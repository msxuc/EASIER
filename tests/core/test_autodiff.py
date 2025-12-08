# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
import os
import tempfile
from typing import Callable, Dict, List, Optional, Sequence, Type, Union
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
from easier.numeric.solver.gmres import GMRES

from ..utils import \
    torchrun_singlenode, assert_tensor_list_equal, \
    when_ngpus_ge_2, mpi_e2e, mpirun_singlenode, \
    import_poisson, import_shallow_water_equation, MESH, POISSON, SW


Poisson = import_poisson()
from assemble_poisson import PoissonInitializer  # type: ignore

ShallowWaterEquation = import_shallow_water_equation()

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
                return (torch.neg(input[idxes] + input_t[idxes]), None)
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
                sort, sort0, sort1, cat_idx, tcat_idx, add, neg, \
                = jvpm.graph_module.graph.nodes
            
            assert tv0.target == operator.getitem and tv0.args[0] == tv
            assert tv1.target == operator.getitem and tv1.args[0] == tv
            assert tcat.target == torch.concat and tcat.args[0] == [tv0, tv1]

            assert sort0.target == operator.getitem and sort0.args == (sort, 0)
            assert sort1.target == operator.getitem and sort1.args == (sort, 1)

            assert cat_idx.target == operator.getitem \
                and cat_idx.args == (cat, sort1)
            assert tcat_idx.target == operator.getitem \
                and tcat_idx.args == (tcat, sort1)
            assert add.target == operator.add \
                and add.args == (cat_idx, tcat_idx)
            
            # Fake node for indication only
            assert neg.target == torch.neg and neg.args == (add,)

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

                b = v0 < v1
                torch.where(b, v0, v1)

        m = M()
        jvpm = esr.jvp(m, [m.v], [])
        jvpm: Jvp

        v, tv, v0, tv0, v1, tv1, cat, tcat, \
            sort, sort0, sort1, tcat_idx, zero, \
            lt, where, twhere \
            = jvpm.graph_module.graph.nodes
        
        assert tv0.target == operator.getitem and tv0.args[0] == tv
        assert tv1.target == operator.getitem and tv1.args[0] == tv

        # torch.jvp expands to `torch.ops.aten.cat.default` API
        assert 'cat' in str(tcat.target) and tcat.args[0] == [tv0, tv1]

        assert sort0.target == operator.getitem and sort0.args == (sort, 0)
        assert sort1.target == operator.getitem and sort1.args == (sort, 1)

        # torch.jvp expands to `torch.ops.aten.gather.default` API
        assert 'gather' in str(tcat_idx.target) \
            and tcat_idx.args == (tcat, 1, sort1)
        
        # the 2nd res item for sort has no tangent
        assert 'zeros_like' in str(zero.target) and zero.args == (sort1,)
    

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
        #
        # EASIER approach
        #
    
        init = PoissonInitializer(POISSON, MESH)
        # _check_op_usage([init])
                
        INIT_VECTOR = torch.rand_like(init.points)

        tpoints = esr.Tensor(INIT_VECTOR, mode='partition')
        jvp = esr.jvp(
            init,
            [init.points], [init.b, init.Ac, init.Af],
            vectors=[tpoints]
        )
        [jvp] = esr.compile([jvp], backend='torch') # type: ignore
        jvp: Jvp

        jvp()

        esr_b = jvp.outputs[0].collect()
        esr_Ac = jvp.outputs[1].collect()
        esr_Af = jvp.outputs[2].collect()
        esr_tb = jvp.products[0].collect()
        esr_tAc = jvp.products[1].collect()
        esr_tAf = jvp.products[2].collect()

        #
        # torch.jvp approach
        #

        init = PoissonInitializer(POISSON, MESH)
        [init] = esr.compile([init], backend='none')

        class _MutableRoot:
            def get_face_norm(self, p0, p1, p2):
                return PoissonInitializer.get_face_norm(self, p0, p1, p2)

        _init = _MutableRoot()
        for k, v in init.__dict__.items():
            setattr(_init, k, v)
        for k, v in init.named_parameters():
            setattr(_init, k, v)
        for k, v in init.named_modules():
            setattr(_init, k, v)

        # All inplace written esr.Tensors must be _inputs_ to torch.jvp().
        input_attrnames = ['points', 'centroid', 'rho', 'b', 'Ac', 'Af']
        output_attrnames = ['b', 'Ac', 'Af']

        def _func(*input_proxies):
            # Although proxies are wrapped on inputs, they are different
            # instances from the `inputs` above, rebind them within the
            # callstack of torch.jvp()
            for n, p in zip(input_attrnames, input_proxies):
                setattr(_init, n, p)
            
            PoissonInitializer.forward(_init)

            return tuple(getattr(_init, n) for n in output_attrnames)
            
        inputs = [getattr(init, n) for n in input_attrnames]
        tangents = [torch.zeros_like(i) for i in inputs]
        tangents[0] = INIT_VECTOR

        torch_outs, torch_tangents = torch.func.jvp(  # type: ignore
            _func, tuple(inputs), tuple(tangents)
        )

        for ep, tp in zip([esr_b, esr_Ac, esr_Af], torch_outs):
            torch.testing.assert_close(ep, tp)
        for et, tt in zip([esr_tb, esr_tAc, esr_tAf], torch_tangents):
            torch.testing.assert_close(et, tt, rtol=1e-6, atol=1e-6)

    def test_smoke__swe_main(self):
        #
        # EASIER approach
        #

        input_attrnames = [
            'x', 'sy', 'bsx', 'bsy', 'alpha',
            # Since esr.jvp currently does not allow overlapping between I/O,
            # the last 3 inputs are effectively outputs too.
            'h', 'uh', 'vh'
        ]

        eqn = ShallowWaterEquation(MESH, SW)
        # _check_op_usage([eqn])

        inputs = [getattr(eqn, n) for n in input_attrnames]

        INIT_VECTOR_VALS = [torch.rand_like(input) for input in inputs]

        vectors = [esr.Tensor(vv, mode='partition') for vv in INIT_VECTOR_VALS]

        jvp = esr.jvp(eqn, inputs, [], vectors=vectors)
        [jvp] = esr.compile([jvp], backend='none') # type: ignore
        jvp: Jvp

        jvp()

        esr_primals = [t.collect() for t in inputs]
        esr_tangents = [vectors[i].collect() for i, t in enumerate(inputs)]

        #
        # torch.jvp approach
        #
        eqn = ShallowWaterEquation(MESH, SW)
        [eqn] = esr.compile([eqn], backend='none') # type: ignore

        # We need such a dict-like container, supporting __getattr__, and
        # don't enforce nn.Parameter field like esr.Module(nn.Module).
        class _MutableRoot:
            def face_reconstruct(self, phi):
                return ShallowWaterEquation.face_reconstruct(self, phi)
            def delta(self, h, uh, vh):
                return ShallowWaterEquation.delta(self, h, uh, vh)

        # All methods, attributes, submods are dispatched to `eqn`
        _eqn = _MutableRoot()
        for k, v in eqn.__dict__.items():
            setattr(_eqn, k, v)
        for k, v in eqn.named_parameters():
            setattr(_eqn, k, v)
        for k, v in eqn.named_modules():
            setattr(_eqn, k, v)
        
        # esr.Tensors for the new eqn instance.
        inputs = [getattr(eqn, n) for n in input_attrnames]

        # torch.jvp() needs explicit return values.
        output_attrnames = ['h', 'uh', 'vh']

        def _func(*input_proxies):
            # Although proxies are wrapped on inputs, they are different
            # instances from the `inputs` above, rebind them within the
            # callstack of torch.jvp()
            for n, p in zip(input_attrnames, input_proxies):
                setattr(_eqn, n, p)
            
            ShallowWaterEquation.forward(_eqn)

            return tuple(getattr(_eqn, n) for n in output_attrnames)
            
        torch_outs, torch_tangents = torch.func.jvp(  # type: ignore
            _func, tuple(inputs), tuple(INIT_VECTOR_VALS)
        )

        for ep, tp in zip(esr_primals[-3:], torch_outs):
            torch.testing.assert_close(ep, tp)
        for et, tt in zip(esr_tangents[-3:], torch_tangents):
            torch.testing.assert_close(et, tt, rtol=1e-6, atol=1e-6)


def _check_op_usage(topmods):
    from easier.core.passes import \
        collectively_initialize_and_validate
    from easier.core.passes.utils import \
        fx_normalize_function_variant_into_kwargs
    _, graphs = collectively_initialize_and_validate(topmods)
    op_ids = set()
    bad_targets = set()
    for g in graphs:
        for n in g.nodes:
            try:
                if n.op == 'call_function':
                    f = n.target
                    d = fx_normalize_function_variant_into_kwargs(
                        f, n.args, n.kwargs
                    )
                    op_ids.add(
                        f.__module__ + '.' + f.__name__ + str(list(d.keys()))
                    )
                elif n.op == 'call_method':
                    f = getattr(torch.ops.aten, n.target)
                    d = fx_normalize_function_variant_into_kwargs(
                        f, n.args, n.kwargs
                    )
                    op_ids.add(n.target + str(list(d.keys())))
            except:
                bad_targets.add(n.target)
    
    print(list(op_ids))
    print(list(bad_targets))


@pytest.mark.usefixtures('dummy_dist_env')
def test_GMRES():
    poisson = Poisson(MESH, POISSON)

    # Poisson.A is LinSys which is not an esr.Module and will be inlined
    gmres = GMRES(poisson.A, poisson.b, poisson.x)
    
    input_attrnames: List[str] = []
    inputs: List[esr.Tensor] = []
    vectors: List[esr.Tensor] = []
    for n, p in gmres.named_parameters(recurse=False):
        if isinstance(p, esr.Tensor):
            t_p = esr.Tensor(
                esr.zeros_like(p),
                mode='partition' if p.is_partition else 'replicate'
            )
            input_attrnames.append(n)
            inputs.append(p)
            vectors.append(t_p)
    
    def _jvp_submod(submod: esr.Module):
        jvpm = esr.jvp(submod, inputs, [], vectors=vectors)
        return jvpm
    
    class UpdateB(esr.Module):
        def __init__(self, B, rnorm):
            super().__init__()

            self.B = B
            self.rnorm = rnorm
        
        def forward(self):
            self.B[0, 0] = self.rnorm
    
    class UpdateH(esr.Module):
        def __init__(self, H, h):
            super().__init__()

            self.H = H
            self.h = h
            self.i = esr.Tensor(
                torch.tensor([0], dtype=torch.int32, device=H.device),
                mode='replicate'
            )
            self.j = esr.Tensor(
                torch.tensor([0], dtype=torch.int32, device=H.device),
                mode='replicate'
            )
        
        def forward(self):
            self.H[self.i, self.j] = self.h


    class UpdateY(esr.Module):
        def __init__(self, H, B, y):
            super().__init__()

            self.H = H
            self.B = B
            self.y = y
            self.j = esr.Tensor(
                torch.tensor([0], dtype=torch.int32, device=H.device),
                mode='replicate'
            )

        def forward(self):
            u, s, vt = torch.linalg.svd(self.H[:self.j + 2], full_matrices=False)
            self.y[:] = vt.transpose(0, 1) @ \
                torch.diag_embed(1 / torch.clamp(s, min=1e-8)) @ \
                u.transpose(0, 1) @ self.B[:self.j + 2]

    update_B = UpdateB(gmres.B, gmres.rnorm)
    update_H = UpdateH(gmres.H, gmres.h)
    update_y = UpdateY(gmres.H, gmres.B, gmres.y)

    # _check_op_usage([sol, update_B, update_H, update_y])
    
    class JvpGMRES(esr.Module):
        def __init__(self):
            super().__init__()
            
            self.jvp_update_rnorm = _jvp_submod(gmres.update_rnorm)
            self.jvp_init = _jvp_submod(gmres.init)

            self.jvp_init_V = _jvp_submod(gmres.init_V)
            self.jvp_init_w = _jvp_submod(gmres.init_w)
            self.jvp_sum_w = _jvp_submod(gmres.sum_w)
            self.jvp_update_w = _jvp_submod(gmres.update_w)
            self.jvp_norm_w = _jvp_submod(gmres.norm_w)
            self.jvp_update_V = _jvp_submod(gmres.update_V)

            self.jvp_update_x = torch.nn.ModuleList(
                _jvp_submod(x) for x in gmres.update_x  # type: ignore
            )

            # Extra esr.Modules
            self.jvp_update_B = _jvp_submod(update_B)
            self.jvp_update_H = _jvp_submod(update_H)
            self.jvp_update_y = _jvp_submod(update_y)

            for n, p in zip(input_attrnames, inputs):
                setattr(self, n, p)

        def _init_w(self, j: int):
            # All these `.i, .j` esr.Tensors have ndim==0,
            # we need to use `fill_()` to set the single element of them.
            self.jvp_init_w.j.fill_(j)
            self.jvp_init_w()

        def _sum_w(self, i: int):
            self.jvp_sum_w.i.fill_(i)
            self.jvp_sum_w()

        def _update_w(self, i: int):
            self.jvp_update_w.i.fill_(i)
            self.jvp_update_w()

        def _update_V(self, i: int):
            self.jvp_update_V.i.fill_(i)
            self.jvp_update_V()

        def _update_y(self, j: int):
            self.jvp_update_y.j.fill_(j)
            self.jvp_update_y()

        def jvp_solve(
            self,
            rtol=1e-5,
            atol: Optional[float] = None,
            maxiter: Optional[int] = None,
            debug_iter: Optional[int] = None
        ):
            name = "JvpGMRES"
            self.jvp_init()
            rtol *= self.bnorm

            tol = max(rtol, atol) if atol else rtol

            iters = 0
            while True:
                self.jvp_update_rnorm()

                if debug_iter is not None and iters % debug_iter == 0:
                    esr.logger.info(
                        f"{name} residual {float(self.rnorm)}"
                        f" at the {iters}-th iteration")

                if (not torch.isnan(self.rnorm) and self.rnorm <= tol) or \
                (maxiter is not None and iters >= maxiter):
                    break
                iters += 1

                # self.B[0, 0] = self.rnorm
                self.jvp_update_B()

                self.jvp_init_V()

                for j in range(self.restart):
                    self._init_w(j)
                    for i in range(j + 1):
                        self._sum_w(i)

                        # self.H[i, j] = self.h
                        self.jvp_update_H.i.fill_(i)
                        self.jvp_update_H.j.fill_(j)
                        self.jvp_update_H()

                        self._update_w(i)

                    self.jvp_norm_w()

                    # self.H[j + 1, j] = self.h
                    self.jvp_update_H.i.fill_(j + 1)
                    self.jvp_update_H.j.fill_(j)
                    self.jvp_update_H()

                    if self.h < 1e-15:
                        break
                    elif j < self.restart - 1:
                        self._update_V(j + 1)

                # u, s, vt = torch.linalg.svd(self.H[:j + 2], full_matrices=False)
                # self.y[:] = vt.transpose(0, 1) @ \
                #     torch.diag_embed(1 / torch.clamp(s, min=1e-8)) @ \
                #     u.transpose(0, 1) @ self.B[:j + 2]
                self.jvp_update_y.j.fill_(j)
                self.jvp_update_y()

                self.jvp_update_x[j]()

    jvp_gmres = JvpGMRES()
    [jvp_gmres] = esr.compile([jvp_gmres], backend='none')  # type: ignore
    jvp_gmres: JvpGMRES

    jvp_gmres.jvp_solve()

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
