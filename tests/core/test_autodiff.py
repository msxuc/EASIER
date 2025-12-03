# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
import os
import tempfile
from typing import Callable, Dict, List, Sequence, Type, Union
from unittest.mock import patch
import pytest
import torch
from torch.fx import Node

import easier as esr
from easier.core.autodiff.autodiff import FxConst, Jvp, JvpTransformer
from easier.core.autodiff.autodiff_rule import \
    DiffRuleBase, Differentiability, tangent_rule_registry, differentiabilities
from easier.core.runtime.metadata import Role, RuntimeTensorMeta
from easier.core.utils import get_random_str

from ..utils import \
    torchrun_singlenode, assert_tensor_list_equal, \
    when_ngpus_ge_2, mpi_e2e, mpirun_singlenode, \
    import_poisson, MESH, POISSON


@pytest.mark.usefixtures('dummy_dist_env')
class TestJvpTransformation:
    
    def test_rule(self):
        # nest args; multi-res
        reg: Dict[Callable, Type[DiffRuleBase]] = dict(tangent_rule_registry)

        class _Cat(DiffRuleBase):
            diffable_params = ['tensors']
            
            def output_meta(self, tensors, dim):
                return RuntimeTensorMeta(Role.DISTRIBUTED, (10, 2), torch.float32)
            
            def jvp(self, tensors, dim, tensors_t):
                return torch.concat(tensors_t, dim)
        reg[torch.concat] = _Cat

        class _Sort(DiffRuleBase):
            needs_result = True
            diffable_params = ['input']

            def output_meta(self, input, dim, descending):
                m1 = RuntimeTensorMeta(Role.DISTRIBUTED, (10, 2), torch.float32)
                m2 = RuntimeTensorMeta(Role.DISTRIBUTED, (10, 2), torch.int64)
                return (m1, m2)
            
            def jvp(self, result, input, dim, descending, input_t):
                (_sort, idxes) = result
                return (torch.neg(input_t[idxes]), None)
        reg[torch.sort] = _Sort

        class M(esr.Module):
            def __init__(self):
                super().__init__()
                self.v = esr.Tensor(esr.zeros([10, 3], dtype=torch.float32), mode='partition')

            def forward(self):
                v0 = self.v[:, 0:1]
                v1 = self.v[:, 1:2]
                vc = torch.concat([v0, v1], dim=1)

                sort, idxes = torch.sort(vc, dim=1)

        with patch(f'{JvpTransformer.__module__}.tangent_rule_registry', new=reg):

            m = M()
            jvpm = esr.jvp(m, [m.v], [])
            jvpm: Jvp

            v, tv, v0, tv0, v1, tv1, cat, tcat, \
                sort, sort0, sort1, tcat_idx, neg, \
                = jvpm.graph_module.graph.nodes
            
            assert tv0.target == operator.getitem and tv0.args[0] == tv
            assert tv1.target == operator.getitem and tv1.args[0] == tv
            assert tcat.target == torch.concat and tcat.args[0] == [tv0, tv1]

            assert sort0.target == operator.getitem and sort0.args == (sort, 0)
            assert sort1.target == operator.getitem and sort1.args == (sort, 1)

            assert tcat_idx.target == operator.getitem \
                and tcat_idx.args == (tcat, sort1)
            
            # Fake node for indication only
            assert neg.target == torch.neg and neg.args == (tcat_idx,)

            # the 2nd res item for sort has no tangent

    def test_torch_jvp_ops(self):
        # nest args; multi-res
        class M(esr.Module):
            def __init__(self):
                super().__init__()
                self.v = esr.Tensor(esr.zeros([10, 3], dtype=torch.float32), mode='partition')
    
            def forward(self):
                v0 = self.v[:, 0:1]
                v1 = self.v[:, 1:2]
                vc = torch.concat([v0, v1], dim=1)

                sort, idxes = torch.sort(vc, dim=1)

        m = M()
        jvpm = esr.jvp(m, [m.v], [])
        jvpm: Jvp

        v, tv, v0, tv0, v1, tv1, cat, tcat, \
            sort, sort0, sort1, tcat_idx, neg, \
            = jvpm.graph_module.graph.nodes
        
        assert tv0.target == operator.getitem and tv0.args[0] == tv
        assert tv1.target == operator.getitem and tv1.args[0] == tv
        assert tcat.target == torch.concat and tcat.args[0] == [tv0, tv1]

        assert sort0.target == operator.getitem and sort0.args == (sort, 0)
        assert sort1.target == operator.getitem and sort1.args == (sort, 1)

        assert tcat_idx.target == operator.getitem \
            and tcat_idx.args == (tcat, sort1)
        
        # Fake node for indication only
        assert neg.target == torch.neg and neg.args == (tcat_idx,)

        # the 2nd res item for sort has no tangent
    

@pytest.mark.usefixtures('dummy_dist_env')
class TestJvp:
    def test_spmv(self):
        nx = 30
        ny = 20

        _ne = nx * ny // 2
        nnz = torch.randint(0, nx * ny, [_ne]).unique(sorted=True)
        ne = nnz.shape[0]

        Ae = torch.rand_like(nnz, dtype=torch.float64)
        x = torch.rand(nx, dtype=torch.float64)
        y = torch.zeros(ny, dtype=torch.float64)

        # matrix form
        A = torch.zeros([ny, nx], dtype=torch.float64)
        A.flatten()[nnz] = Ae

        class SpMV(esr.Module):
            def __init__(self):
                super().__init__()

                p = torch.randperm(ne)
                nnz2 = nnz[p]
                s_idx = nnz2 % nx
                r_idx = nnz2 // nx

                self.Ae = esr.Tensor(Ae[p], mode='partition')
                self.selector = esr.Selector(s_idx)
                self.reducer = esr.Reducer(r_idx, ny)

                self.x = esr.Tensor(x, mode='partition')
                self.y = esr.Tensor(y, mode='partition')

            def forward(self):
                y = self.reducer(
                    self.selector(self.x) * self.Ae
                )
                self.y[:] = y
        
        tangent_x_datasrc = torch.rand_like(x)

        raw = SpMV()
        tx = esr.Tensor(tangent_x_datasrc, mode='partition')
        jvp = esr.jvp(raw, [raw.x], [raw.y, raw.Ae], vectors=[tx])
        [jvp] = esr.compile([jvp], backend='torch') # type: ignore
        jvp: Jvp

        jvp()

        esr_y = jvp.outputs[0].collect()
        esr_ty = jvp.products[0].collect()

        # classical mv and jvp grounding
        torch_y, torch_ty = torch.func.jvp(  # type: ignore
            torch.mv,
            primals=(A, x),
            tangents=(torch.zeros_like(A), tangent_x_datasrc)
        )

        torch.testing.assert_close(esr_y, torch_y)
        torch.testing.assert_close(esr_ty, torch_ty)


    def test_smoke__assemble_poisson(self):
        pass

@pytest.mark.parametrize('dev_type', [
    'cpu',
    pytest.param('cuda', marks=when_ngpus_ge_2)
])
def test_dump_jvp(dev_type: str):
    dumpdir = os.path.join(tempfile.gettempdir(), "easier", "tests",
                               get_random_str())
        
    torch.manual_seed(2345)
    model_dev = torch.device(dev_type)

    m = Model(3, model_dev)  # type: ignore

    jm1, = esr.compile([m], 'torch', partition_mode='evenly')  # type: ignore
    esr.dump([jm1], dumpdir)
    jm1: Model

    torch.manual_seed(2345)
    m = Model(3, model_dev)  # type: ignore
    jm2, = esr.compile(
        [m], 'torch', load_dir=dumpdir, partition_mode='evenly'  # type: ignore
    )
    jm2: Model

    _equal_jitted_selector(jm1.selector_src, jm2.selector_src)
    _equal_jitted_selector(jm1.selector_dst, jm2.selector_dst)
    _equal_jitted_selector(
        getattr(jm1, 'csr_selector0reducer_src'),
        getattr(jm2, 'csr_selector0reducer_src')
    )
    _equal_jitted_reducer(jm1.reducer_src, jm2.reducer_src)
    _equal_jitted_reducer(jm1.reducer_dst, jm2.reducer_dst)

    _equal_jitted_selector(
        getattr(jm1, 'reordering_selector0reducer_dst'),
        getattr(jm2, 'reordering_selector0reducer_dst'),
    )
    _equal_jitted_selector(
        getattr(jm1, 'reordering_selector1reducer_src'),
        getattr(jm2, 'reordering_selector1reducer_src'),
    )
