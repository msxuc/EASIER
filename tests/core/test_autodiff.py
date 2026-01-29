# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
import os
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type
from unittest.mock import patch
import pytest
import torch
from torch.fx import Node

import easier as esr
from easier.core.runtime.data_loader import InMemoryTensorLoader
from easier.core.autodiff.jvp import Jvp, JvpTransformer
from easier.core.autodiff.vjp import Vjp, VjpTransformer
from easier.core.autodiff.autodiff_rule import \
    DiffRuleBase, Differentiability, \
    diff_rule_registry, differentiabilities
from easier.core.runtime.metadata import Role, RuntimeTensorMeta
from easier.core.utils import get_random_str
from easier.core.passes.utils import SubmodNameAllocator, get_easier_objects
from easier.numeric.solver import CG, GMRES

from ..utils import \
    torchrun_singlenode, assert_tensor_list_equal, \
    when_ngpus_ge_2, mpi_e2e, mpirun_singlenode, \
    import_poisson, import_shallow_water_equation, \
    MESH_100, POISSON_100, SW_100, \
    MESH_30, POISSON_30, \
    linsys_to_mat


import_poisson()  # activate Python search paths only
from assemble_poisson import PoissonInitializer  # type: ignore

ShallowWaterEquation = import_shallow_water_equation()

@pytest.mark.usefixtures('dummy_dist_env')
class TestJvpTransformation:
    
    def test_rule(self):
        # nest args; multi-res
        reg: Dict[Callable, Type[DiffRuleBase]] = dict(diff_rule_registry)

        class _Cat(DiffRuleBase):
            diffable_params = ['tensors']
            
            def output_meta(self, tensors, dim):
                return RuntimeTensorMeta(
                    Role.DISTRIBUTED, (10, 2), torch.float32
                )
            
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
                self.v = esr.Tensor(
                    esr.zeros([10, 3], dtype=torch.float32), mode='partition'
                )

            def forward(self):
                v0 = self.v[:, 0:1]
                v1 = self.v[:, 1:2]
                vc = torch.concat([v0, v1], dim=1)

                sort, idxes = torch.sort(vc, dim=1)

        with patch(f'{JvpTransformer.__module__}.diff_rule_registry', new=reg):

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
    
    def test_no_jvp_ops(self):
        # nest args; multi-res
        class M(esr.Module):
            def __init__(self):
                super().__init__()
                self.x = esr.Tensor(
                    esr.zeros([10, 3], dtype=torch.float32), mode='partition'
                )
                self.v = esr.Tensor(
                    esr.zeros([10, 3], dtype=torch.float32), mode='partition'
                )
    
            def forward(self):
                v0 = self.v[:, 0:1]
                v1 = self.v[:, 1:2]
                vc = torch.concat([v0, v1], dim=1)

                sort, idxes = torch.sort(vc, dim=1)

                b = v0 < v1
                torch.where(b, v0, v1)

        m = M()
        jvpm = esr.jvp(m, [m.x], [])
        jvpm: Jvp

    def test_torch_jvp_ops(self):
        # nest args; multi-res
        class M(esr.Module):
            def __init__(self):
                super().__init__()
                self.v = esr.Tensor(
                    esr.zeros([10, 3], dtype=torch.float32), mode='partition'
                )
    
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
        _test_jvp(
            lambda: PoissonInitializer(POISSON_100, MESH_100),
            ['get_face_norm'],
            ['points'],
            ['b', 'Ac', 'Af'],
        )

    def test_smoke__swe_main(self):
        _test_jvp(
            lambda: ShallowWaterEquation(MESH_100, SW_100),
            ['face_reconstruct', 'delta'],
            [
                'x',
                'sy',
                'bsx', 'bsy',
                'alpha',
            ],
            ['h', 'uh', 'vh'],
            rtol=1e-5, atol=1e-6
        )


@pytest.mark.usefixtures('dummy_dist_env')
class TestVjpTransformer:

    def test_rule(self):
        # nest args; multi-res
        reg: Dict[Callable, Type[DiffRuleBase]] = dict(diff_rule_registry)

        class _Cat(DiffRuleBase):
            diffable_params = ['tensors']
            
            def output_meta(self, tensors, dim):
                return RuntimeTensorMeta(
                    Role.DISTRIBUTED, (10, 2), torch.float32
                )
            
            def vjp(self, tensors, dim, cotangent):
                n = len(tensors)
                return cotangent.chunk(n, dim=dim)
        reg[torch.concat] = _Cat

        class _Aminmax(DiffRuleBase):
            diffable_params = ['input']
            output_differentiability = [True, True]

            def output_meta(self, input, dim, keepdim):
                item_meta = RuntimeTensorMeta(
                    Role.DISTRIBUTED, (10,), torch.float32
                )
                return [item_meta, item_meta]
        reg[torch.aminmax] = _Aminmax

        class M(esr.Module):
            def __init__(self):
                super().__init__()
                self.v = esr.Tensor(
                    esr.zeros([10, 3], dtype=torch.float32), mode='partition'
                )
                self.amin_out = esr.Tensor(
                    esr.zeros([10], dtype=torch.float32), mode='partition'
                )

            def forward(self):
                v0 = self.v[:, 0:1]
                v1 = self.v[:, 1:2]
                vc = torch.concat([v0, v1], dim=1)

                amin, amax = torch.aminmax(vc, dim=1)

                self.amin_out[:] = amin

        with patch(f'{VjpTransformer.__module__}.diff_rule_registry', new=reg):

            m = M()
            vjpm = esr.vjp(m, [m.v], [])
            vjpm: Jvp

    def test_no_vjp_ops(self):
        # nest args; multi-res
        class M(esr.Module):
            def __init__(self):
                super().__init__()
                self.not_used = esr.Tensor(
                    esr.zeros([10, 3], dtype=torch.float32), mode='partition'
                )
                self.v = esr.Tensor(
                    esr.zeros([10, 3], dtype=torch.float32), mode='partition'
                )
    
            def forward(self):
                v0 = self.v[:, 0:1]
                v1 = self.v[:, 1:2]
                vc = torch.concat([v0, v1], dim=1)

                sort, idxes = torch.sort(vc, dim=1)

                b = v0 < v1
                torch.where(b, v0, v1)

        m = M()
        jvpm = esr.jvp(m, [m.not_used], [])
        jvpm: Jvp

    def test_torch_vjp_ops(self):
        # nest args; multi-res
        class M(esr.Module):
            def __init__(self):
                super().__init__()
                self.v = esr.Tensor(
                    esr.zeros([10, 3], dtype=torch.float32), mode='partition'
                )
    
            def forward(self):
                v0 = self.v[:, 0:1]
                v1 = self.v[:, 1:2]
                vc = torch.concat([v0, v1], dim=1)

                v0, v1 = torch.aminmax(vc, dim=1)

                b = v0 < v1
                torch.where(b, v0, v1)

        m = M()
        jvpm = esr.jvp(m, [m.v], [])
        jvpm: Jvp

@pytest.mark.usefixtures('dummy_dist_env')
class TestVjp:
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
        
        cot_y_datasrc = torch.rand_like(y)

        raw = SpMV()
        cot_y = esr.Tensor(cot_y_datasrc, mode='partition')
        vjp = esr.vjp(raw, [raw.x], [raw.y], vectors=[cot_y]) # type: ignore
        [vjp] = esr.compile([vjp], backend='torch') # type: ignore
        vjp: Vjp

        vjp()

        esr_y = vjp.outputs[0].collect()
        esr_gradx = vjp.products[0].collect()

        # classical mv and jvp grounding
        torch_y, vjpfunc = torch.func.vjp( # type: ignore
            torch.mv,
            A, x
        )
        torch_gradA, torch_gradx = vjpfunc(cot_y_datasrc)

        torch.testing.assert_close(esr_y, torch_y)
        torch.testing.assert_close(esr_gradx, torch_gradx)


def _test_jvp(
    ctor: Callable[[], esr.Module],
    redirected_methods: List[str] = [],
    input_attrnames: List[str] | None = None,
    output_attrnames: List[str] = [],  # not overlapping
    # Otherwise random vectors/tangents are used.
    optional_vectors: Dict[str, torch.Tensor] = {},
    rtol=None, atol=None,
    randomize_initial_inputs=False
):
    module = ctor()
    # _check_op_usage([module])

    if not input_attrnames:
        input_attrnames = []
        for n, p in module.named_parameters():
            if '.' not in n and p.dtype.is_floating_point:
                input_attrnames.append(n)


    #
    # Prepare constant torch.Tensor initial tangents for inputs
    #
    _merged_inputs = [
        getattr(module, n) for n in input_attrnames + output_attrnames
    ]
    INIT_RANDOM_INPUT = list(map(torch.rand_like, _merged_inputs))
    INIT_VECTORS = list(map(torch.rand_like, _merged_inputs))

    for i, n in enumerate(input_attrnames):
        if n in optional_vectors:
            INIT_VECTORS[i] = optional_vectors[n]

    #
    # EASIER approach
    #
    esr_inputs = [
        getattr(module, n) for n in input_attrnames + output_attrnames
    ]
    if randomize_initial_inputs:
        for esr_i, r_i in zip(esr_inputs, INIT_RANDOM_INPUT):
            esr_i: esr.Tensor
            esr_i.easier_data_loader = InMemoryTensorLoader(r_i)

    esr_vectors: List[esr.Tensor] = []
    for i, v in zip(esr_inputs, INIT_VECTORS):
        esr_vectors.append(esr.Tensor(
            v, mode='partition' if i.is_partition else 'replicate'
        ))
    
    jvp = esr.jvp(module, esr_inputs, [], esr_vectors)
    [jvp] = esr.compile([jvp], backend='torch')  # type: ignore
    jvp: Jvp

    jvp()

    esr_primals = [t.collect() for t in jvp.inputs]
    esr_tangents = [t.collect() for t in jvp.vectors]

    #
    # torch.jvp approach
    #
    module = ctor()

    esr_inputs = [
        getattr(module, n) for n in input_attrnames + output_attrnames
    ]
    if randomize_initial_inputs:
        for esr_i, r_i in zip(esr_inputs, INIT_RANDOM_INPUT):
            esr_i: esr.Tensor
            esr_i.easier_data_loader = InMemoryTensorLoader(r_i)

    vectors = dict(zip(input_attrnames + output_attrnames, INIT_VECTORS))

    torch_outs, torch_tangents = _run_torch_jvp(
        module, redirected_methods, vectors,
        input_attrnames + output_attrnames
    )

    for ep, tp in zip(esr_primals, torch_outs):
        torch.testing.assert_close(ep, tp)
    for et, tt in zip(esr_tangents, torch_tangents):
        torch.testing.assert_close(et, tt, rtol=rtol, atol=atol)

    



def _run_torch_jvp(
    module: esr.Module, redirected_methods: List[str],
    vectors: Dict[str, torch.Tensor],
    output_attrnames: List[str]
) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:

    [module] = esr.compile([module], backend='none')

    # We need such a dict-like container, supporting __getattr__, and
    # don't enforce nn.Parameter field like esr.Module(nn.Module).
    class _MutableRoot:
        pass

    raw_cls = module.__class__

    for method in redirected_methods:

        # Introduce an explicit var binding to capture loop var 'method'
        def _set_method(method: str):
            def _dispatch(*args, **kwargs):
                return getattr(raw_cls, method)(*args, **kwargs)
            setattr(_MutableRoot, method, _dispatch)

        _set_method(method)
    
    all_params: Dict[str, torch.Tensor] = {}

    # All methods, attributes, submods are dispatched to `eqn`
    root = _MutableRoot()
    for k, v in module.__dict__.items():
        setattr(root, k, v)
    for k, v in module.named_parameters():
        setattr(root, k, v)

        if '.' not in k and v.dtype.is_floating_point:
            all_params[k] = v

    for k, v in module.named_modules():
        setattr(root, k, v)

    def _func(*param_proxies):
        # Although proxies are wrapped on inputs, they are different
        # instances from the `inputs` above, rebind them within the
        # callstack of torch.jvp()
        for n, p in zip(all_params.keys(), param_proxies):
            setattr(root, n, p)
        
        raw_cls.forward(root)  # type: ignore

        return tuple(getattr(root, n) for n in output_attrnames)
    
    init_tangents = {
        k: vectors.get(k, torch.zeros_like(v))
        for k, v in all_params.items()
    }
        
    torch_outs, torch_tangents = torch.func.jvp(  # type: ignore
        _func, tuple(all_params.values()), tuple(init_tangents.values())
    )

    return torch_outs, torch_tangents



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


def _dump_jvp_graph_module(topmods):
    """
    Usage:
    -   compile with 'none' backend
    -   call this after compile
    -   set breakpoint in the local function _jvp_fw below.
    """
    import importlib

    dump_dir = os.path.join(tempfile.gettempdir(), "easier", "jvp")
    dump_dir = os.path.expanduser(dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    if dump_dir not in sys.path:
        sys.path.append(dump_dir)

    for jvpm, hint_name in get_easier_objects(topmods).items():
        if not isinstance(jvpm, Jvp):
            continue

        # code looks like:
        # ```
        # torch.fx._symbolic_trace.wrap("easier_core_module_sum")
        # torch.fx._symbolic_trace.wrap("easier_core_module_norm")
        #
        # def forward(self):
        #    p = self.p
        #    vectors_4 = getattr(self.vectors, "4")
        #    ...
        #    a_reducer = self.A.reducer(mul);  mul = other = None
        #    ...
        # ```

        code = jvpm.graph_module.graph.python_code('self').src
        fw_start_pos = code.index('def forward(')

        fw_lines = []
        for src_fw_line in code[fw_start_pos:].splitlines():
            # If no GC parts, the pos is -1
            gc_start_pos = src_fw_line.find(';')
            if gc_start_pos > 0:
                src_fw_line = src_fw_line[:gc_start_pos]

            fw_lines.append('    ' + src_fw_line + '\n')

        # fx.Codegen offers little fine-granularity utils to control codegen
        # so we cannot add hint for primal-tangent-raw mapping.
        # TODO custom codegen by ourselves
        fname = SubmodNameAllocator('').purify_attr_name(hint_name[0])
        fpath = os.path.join(dump_dir, f'{fname}.py')
        with open(fpath, 'w') as fs:
            fs.write(f"""
import torch
import easier

easier_core_module_sum = easier.core.module.sum
easier_core_module_norm = easier.core.module.norm

class {fname}:
""")
            fs.writelines(fw_lines)

        importlib.invalidate_caches()
        # otherwise immediately import_module may fail with ModuleNotFound

        fmod = importlib.import_module(fname)
        cls = getattr(fmod, fname)

        def _set_fw(jvpm: Jvp, cls):

            def _jvp_fw(self: Jvp):
                for p in self.products:
                    p.zero_()
                
                #
                # Debugger insert breakpoint at the line below, then step-into.
                #

                cls.forward(self.graph_module)

            jvpm.forward = _jvp_fw.__get__(jvpm)
        
        _set_fw(jvpm, cls)  # capture iter vars `jvpm, cls`
        

@pytest.mark.skip('tangent is not numerically stable')
@pytest.mark.usefixtures('dummy_dist_env')
def test_CG():
    Poisson30 = import_poisson(30)

    # CG.A is LinSys which is not an esr.Module and will be inlined
    def _make_cg():
        poisson = Poisson30(MESH_30, POISSON_30)
        cg = CG(poisson.A, poisson.b, poisson.x)
        return cg


    tol=1e-9  # preciser for small mesh size 30 

    #
    # Invoke CG
    #
    cg = _make_cg()
    [cg] = esr.compile([cg], backend='none')  # type: ignore
    cg: CG

    b = cg.b.collect()
    NV = cg.x.shape[0]

    from easier.numeric.linsys import Linsys
    A: Linsys = cg.A  # type: ignore
    M = linsys_to_mat(NV, NV, A.selector.idx, A.reducer.idx, A.Ae, A.Av)
    M_inv =  torch.inverse(M)
    real_x = M_inv @ cg.b

    cg.solve(atol=tol, maxiter=1000, debug_iter=10)

    solved_x = cg.x.collect()
    torch.testing.assert_close(solved_x, real_x)


    #
    # Invoke JVP CG
    #
    cg = _make_cg()

    input_attrnames: List[str] = []
    inputs: List[esr.Tensor] = []
    vectors: Dict[str, esr.Tensor] = {}
    for n, p in cg.named_parameters(recurse=False):
        if isinstance(p, esr.Tensor):
            input_attrnames.append(n)
            inputs.append(p)

            t_p = esr.Tensor(
                esr.zeros_like(p),
                mode='partition' if p.is_partition else 'replicate'
            )
            vectors[n] = t_p


    INIT_T_B = torch.rand_like(vectors['b'])
    INIT_T_B = torch.nn.functional.normalize(INIT_T_B, dim=0)
    vectors['b'] = esr.Tensor(INIT_T_B, mode='partition')

    def _jvp_submod(submod: esr.Module):
        jvpm = esr.jvp(submod, inputs, [], vectors=list(vectors.values()))
        return jvpm

    class JvpCG(esr.Module):
        def __init__(self):
            super().__init__()

            self.jvp_init = _jvp_submod(cg.init)
            self.jvp_step = _jvp_submod(cg.step)
            self.jvp_update = _jvp_submod(cg.update)

            for n, p in zip(input_attrnames, inputs):
                setattr(self, n, p)
        
        def jvp_solve(
            self,
            rtol: float = 1e-5,
            atol: Optional[float] = None,
            maxiter: Optional[int] = None,
            debug_iter: Optional[int] = None
        ) -> Dict[str, Any]:
            name = self.__class__.__name__
            self.jvp_init()
            rtol *= self.bnorm
            tol = max(rtol, atol) if atol else rtol

            iters = 0
            while True:
                self.jvp_step()

                if debug_iter is not None and iters % debug_iter == 0:
                    esr.logger.info(
                        f"{name} residual {float(self.rnorm)}"
                        f" at the {iters}-th iteration")

                if (not torch.isnan(self.rnorm) and self.rnorm <= tol) or \
                (maxiter is not None and iters >= maxiter):
                    break
                iters += 1

                self.jvp_update()

            esr.logger.info(
                f"{name} solver completed with residual {float(self.rnorm)}" +
                f" at the {iters}-th iteration")

            return {'residual': float(self.rnorm), 'iters': iters}

    jvp_cg = JvpCG()
    [jvp_cg] = esr.compile([jvp_cg], backend='none')  # type: ignore
    jvp_cg: JvpCG

    # _dump_jvp_graph_module([jvp_cg])

    jvp_cg.jvp_solve(atol=tol, maxiter=1000, debug_iter=10)

    esr_x = jvp_cg.x.collect()  # type: ignore
    torch.testing.assert_close(esr_x, real_x)

    esr_tx = vectors['x'].collect()

    torch_tx = M_inv @ INIT_T_B
    torch.testing.assert_close(esr_tx, torch_tx)


@pytest.mark.usefixtures('dummy_dist_env')
@pytest.mark.parametrize(
    'test_component',
    [
        pytest.param(
            True, id='component'
        ),
        pytest.param(
            False, id='solver', marks=pytest.mark.skip(
                'tangent is not numerically stable'
            )
        ),
    ]
) 
def test_GMRES(test_component: bool):
    Poisson30 = import_poisson(30)

    # Poisson.A is LinSys which is not an esr.Module and will be inlined
    def _make_gmres():
        poisson = Poisson30(MESH_30, POISSON_30)
        gmres = GMRES(poisson.A, poisson.b, poisson.x)
        return gmres

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

    #
    # test if Jvp of individual esr.Module compoenent works well
    #
    if test_component:
        class _GmresCompCtor:
            def __getattribute__(self, name: str):
                def _make():
                    gmres = _make_gmres()
                    return getattr(gmres, name)
                return _make
        _gmres = _GmresCompCtor()
        # We need somehow to create new component esr.Module instance because
        # each Module/Tensor can be compiled only once.

        _test_jvp(_gmres.update_rnorm, randomize_initial_inputs=True)
        _test_jvp(_gmres.init, randomize_initial_inputs=True)

        _test_jvp(_gmres.init_V, randomize_initial_inputs=True)
        _test_jvp(_gmres.init_w, randomize_initial_inputs=True)
        _test_jvp(_gmres.sum_w, randomize_initial_inputs=True)
        _test_jvp(_gmres.update_w, randomize_initial_inputs=True)
        _test_jvp(_gmres.norm_w, randomize_initial_inputs=True)
        _test_jvp(_gmres.update_V, randomize_initial_inputs=True)

        for i in range(20):
            _test_jvp(lambda: _gmres.update_x()[i], randomize_initial_inputs=True)

        _test_jvp(
            lambda: UpdateB(_gmres.B(), _gmres.rnorm()),
            randomize_initial_inputs=True
        )
        _test_jvp(
            lambda: UpdateH(_gmres.H(), _gmres.h()),
            randomize_initial_inputs=True
        )
        _test_jvp(
            lambda: UpdateY(_gmres.H(), _gmres.B(), _gmres.y()),
            randomize_initial_inputs=True
        )
        return
    

    tol=1e-9  # preciser for small mesh size 30 
    
    # #
    # # Invoke raw GMRES
    # #
    # gmres = _make_gmres()
    # [gmres] = esr.compile([gmres], backend='none')  # type: ignore
    # gmres: GMRES

    # b = gmres.b.collect()
    # NV = gmres.x.shape[0]

    # from easier.numeric.linsys import Linsys
    # A: Linsys = gmres.A  # type: ignore
    # M = linsys_to_mat(NV, NV, A.selector.idx, A.reducer.idx, A.Ae, A.Av)
    # M_inv =  torch.inverse(M)
    # real_x = M_inv @ gmres.b

    # gmres.solve(atol=tol, maxiter=1000, debug_iter=10)

    # solved_x = gmres.x.collect()
    # torch.testing.assert_close(solved_x, real_x)
    

    #
    # Invoke JVP GMRES
    #
    gmres = _make_gmres()

    input_attrnames: List[str] = []
    inputs: List[esr.Tensor] = []
    vectors: Dict[str, esr.Tensor] = {}
    for n, p in gmres.named_parameters(recurse=False):
        if isinstance(p, esr.Tensor):
            input_attrnames.append(n)
            inputs.append(p)

            t_p = esr.Tensor(
                esr.zeros_like(p),
                mode='partition' if p.is_partition else 'replicate'
            )
            vectors[n] = t_p


    INIT_T_B = torch.rand_like(vectors['b'])
    INIT_T_B = torch.nn.functional.normalize(INIT_T_B, dim=0)
    vectors['b'] = esr.Tensor(INIT_T_B, mode='partition')

    def _jvp_submod(submod: esr.Module):
        jvpm = esr.jvp(submod, inputs, [], vectors=list(vectors.values()))
        return jvpm
    

    update_B = UpdateB(gmres.B, gmres.rnorm)
    update_H = UpdateH(gmres.H, gmres.h)
    update_y = UpdateY(gmres.H, gmres.B, gmres.y)

    # _check_op_usage([gmres, update_B, update_H, update_y])

    
    class JvpGMRES(esr.Module):
        def __init__(self, restart: int):
            super().__init__()

            self.restart = restart
            
            self.jvp_update_rnorm = _jvp_submod(gmres.update_rnorm)
            self.jvp_init = _jvp_submod(gmres.init)

            self.jvp_init_V = _jvp_submod(gmres.init_V)
            self.jvp_init_w = _jvp_submod(gmres.init_w)
            self.jvp_sum_w = _jvp_submod(gmres.sum_w)
            self.jvp_update_w = _jvp_submod(gmres.update_w)
            self.jvp_norm_w = _jvp_submod(gmres.norm_w)
            self.jvp_update_V = _jvp_submod(gmres.update_V)

            self.jvp_update_x = torch.nn.ModuleList(
                _jvp_submod(up) for up in gmres.update_x  # type: ignore
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
    
    jvp_gmres = JvpGMRES(gmres.restart)
    [jvp_gmres] = esr.compile([jvp_gmres], backend='none')  # type: ignore
    jvp_gmres: JvpGMRES

    _dump_jvp_graph_module([jvp_gmres])

    jvp_gmres.jvp_solve(atol=tol, maxiter=1000, debug_iter=10)

    esr_x = jvp_gmres.x.collect()  # type: ignore
    torch.testing.assert_close(esr_x, real_x)

    esr_tx = vectors['x'].collect()

    torch_tx = M_inv @ INIT_T_B
    torch.testing.assert_close(esr_tx, torch_tx)


