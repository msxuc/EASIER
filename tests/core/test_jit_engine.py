# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, cast
import pytest
import torch

from torch.fx.node import Node

from easier.core.jit import EasierTracer, _fully_load_data_backend_none

from easier.core.module import Selector, Reducer, Tensor
from easier.core.runtime.jit_engine.jit_engine import JitEngine
from easier.core.runtime.metadata import Role, get_node_meta, RuntimeTensorMeta
from easier.core.passes.utils import FX
import easier as esr


def _assert_distributed(node: Node, batch_size: int, subshape=None):
    meta = get_node_meta(node)
    assert isinstance(meta, RuntimeTensorMeta)
    assert meta.role == Role.DISTRIBUTED
    assert meta.shape[0] == batch_size

    if subshape is not None:
        assert meta.shape[1:] == tuple(subshape)

    return meta


def _assert_replica(node: Node, shape=None):
    meta = get_node_meta(node)
    assert isinstance(meta, RuntimeTensorMeta)
    assert meta.role == Role.REPLICATED

    if shape is not None:
        assert meta.shape == tuple(shape)

    return meta


@pytest.mark.usefixtures('dummy_dist_env')
class TestMetadataPropagation:
    """
    Cases that TensorMeta.view_src being static and fixed are tested in
    passes/test_data_dependency_analysis.py

    TODO in the future we may allow metadata change between runs.
    """

    def test_easier_primitives(self):
        class M(esr.Module):
            def __init__(self):
                super().__init__()
                self.s = Selector(torch.LongTensor([1, 2, 3, 4]))
                self.r = Reducer(torch.LongTensor([6, 7, 8, 9]), 22)
                self.v = Tensor(torch.zeros([11, 3]), mode='partition')

            def forward(self):
                e = self.s(self.v)
                v2 = self.r(e)

        m = M()
        graph = EasierTracer().trace(m)

        _fully_load_data_backend_none([m], 'cpu')
        engine = JitEngine(m, graph)
        engine.forward()

        getattr_v, call_s, call_r, output = graph.nodes

        _assert_distributed(getattr_v, 11, [3])
        _assert_distributed(call_s, 4, [3])
        _assert_distributed(call_r, 22, [3])
        _assert_replica(output)

    def test_getsetitem(self):
        def _case(f) -> List[Node]:
            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v = Tensor(torch.zeros([11, 2, 3, 4]),
                                    mode='partition')
                    self.t = Tensor(torch.zeros([5, 6, 7], dtype=torch.int32),
                                    mode='replicate')

                def forward(self):
                    f(self.v, self.t)

            m = M()
            graph = EasierTracer().trace(m)

            _fully_load_data_backend_none([m], 'cpu')
            engine = JitEngine(m, graph)
            engine.forward()

            # nodes: v, t, ..., out
            return list(graph.nodes)[2:-1]

        slice1, = _case(lambda v, t: v[:])
        _assert_distributed(slice1, 11)

        slice2, = _case(lambda v, t: v[:, None, :, 1, 2:4])
        _assert_distributed(slice2, 11)

        def _bc_t_113_into_23(v, t):
            t[0, :2, :3] = t[:1, :1, :3]
        gett3, sett3, = _case(_bc_t_113_into_23)
        _assert_replica(gett3)
        _assert_replica(sett3)

        def _case4(v, t):
            t2 = t + 2  # out-of-index, but ok for metadata
            v[:, None, 0:2, ..., None, t2]
        calct2, slice4, = _case(_case4)
        _assert_replica(calct2, [5, 6, 7])
        _assert_distributed(slice4, 11)

        def _case5(v, t):
            v[:] = v * 2
        mul, setv, = _case(_case5)
        _assert_distributed(mul, 11)
        _assert_distributed(setv, 11)

    def test_arith_ops(self):

        def _case(f) -> List[Node]:
            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v1 = Tensor(torch.zeros([11, 2, 3, 4]),
                                     mode='partition')
                    self.v2 = Tensor(torch.zeros([11, 1, 1, 1]),
                                     mode='partition')

                def forward(self):
                    f(self.v1, self.v2)

            m = M()
            graph = EasierTracer().trace(m)

            _fully_load_data_backend_none([m], 'cpu')
            engine = JitEngine(m, graph)
            engine.forward()

            # nodes: v1, v2, ..., out
            return list(graph.nodes)[2:-1]

        add, = _case(torch.add)
        _assert_distributed(add, 11)

        pow, = _case(torch.pow)
        _assert_distributed(pow, 11)

    def test_reduction(self):
        def _case(f) -> List[Node]:

            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v = Tensor(torch.zeros(
                        [11, 2, 3, 4], dtype=torch.int32), mode='partition')

                def forward(self):
                    f(self.v)

            m = M()
            graph = EasierTracer().trace(m)

            _fully_load_data_backend_none([m], 'cpu')
            engine = JitEngine(m, graph)
            engine.forward()

            # nodes: v, ..., out
            return list(graph.nodes)[1:-1]

        sum_full, sum_0rank = _case(lambda x: esr.sum(x).sum())
        _assert_replica(sum_full, [1, 2, 3, 4])
        _assert_replica(sum_0rank, [])

    def test_overloads(self):
        def _case(fv=lambda x: x, ft=lambda x: x) -> List[Node]:

            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v = esr.Tensor(torch.zeros(
                        [11, 2, 3, 4], dtype=torch.int32), mode='partition')
                    self.t = esr.Tensor(torch.zeros(
                        [11, 2, 3, 4], dtype=torch.int32), mode='replicate')

                def forward(self):
                    v = self.v
                    t = self.t
                    fv(v)
                    ft(t)

            m = M()
            graph = EasierTracer().trace(m)

            _fully_load_data_backend_none([m], 'cpu')
            engine = JitEngine(m, graph)
            engine.forward()

            # nodes: v, t, ..., out
            return list(graph.nodes)[2:-1]

        # NOTE FX hooks functions when users are accessing a global variable
        # (here the variable is the function name)
        # so we need the lambda expression to ensure such an access pattern,
        # instead of writing `fv=esr.sum` which is merely a function object.
        sum_full, = _case(fv=lambda x: esr.sum(x))
        _assert_replica(sum_full)

        sum_full, = _case(ft=lambda x: torch.sum(x))
        _assert_replica(sum_full)

        sum_dims_v, sum_dims_t, = _case(
            fv=lambda x: x.sum([1, 2]),
            ft=lambda x: x.sum([1, 2])
        )
        _assert_distributed(sum_dims_v, 11)
        _assert_replica(sum_dims_t)

        sum_dims_v, sum_dims_t, = _case(
            fv=lambda x: x.sum([1, 2], keepdim=True),
            ft=lambda x: x.sum([1, 2], keepdim=True)
        )
        _assert_distributed(sum_dims_v, 11)
        _assert_replica(sum_dims_t)

    def test_where(self):

        def _case(f) -> List[Node]:
            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v1 = Tensor(torch.zeros([11, 2, 3, 4]),
                                     mode='partition')
                    self.v2 = Tensor(torch.zeros([11, 1, 1, 1]),
                                     mode='partition')

                def forward(self):
                    f(self.v1 == self.v2, self.v1, self.v2)

            m = M()
            graph = EasierTracer().trace(m)

            _fully_load_data_backend_none([m], 'cpu')
            engine = JitEngine(m, graph)
            engine.forward()

            # nodes: v1, v2, ..., out
            return list(graph.nodes)[:-1]

        v1, v2, eq, where = _case(torch.where)
        assert where.op == FX.CALL_FUNCTION and where.target == torch.where
        assert where.args == (eq, v1, v2)
        _assert_distributed(eq, 11)
        _assert_distributed(where, 11)

        v1, v2, eq, where = _case(lambda c, x, y: x.where(c, y))
        assert where.op == FX.CALL_METHOD and where.target == 'where'
        assert where.args == (v1, eq, v2)
        _assert_distributed(eq, 11)
        _assert_distributed(where, 11)

    def test_matmul(self):

        def _case(f) -> List[Node]:
            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v1 = Tensor(torch.zeros([11, 8, 3, 4]),
                                     mode='partition')
                    self.v2 = Tensor(torch.zeros([11, 1, 4, 5]),
                                     mode='partition')
                    self.t1 = Tensor(torch.zeros(3, 4), mode='replicate')
                    self.t2 = Tensor(torch.zeros(4, 5), mode='replicate')

                def forward(self):
                    f(self.v1, self.v2, self.t1, self.t2)

            m = M()
            graph = EasierTracer().trace(m)

            _fully_load_data_backend_none([m], 'cpu')
            engine = JitEngine(m, graph)
            engine.forward()

            # nodes: v1, v2, t1, t2, ..., out
            return list(graph.nodes)[4:-1]

        mm, = _case(lambda v, w, t, s: v @ w)
        _assert_distributed(mm, 11)

        t0, mm, = _case(lambda v, w, t, s: v @ t[0])
        _assert_distributed(mm, 11)

        t0, mm, = _case(lambda v, w, t, s: t[0] @ w)
        _assert_distributed(mm, 11)

    def test_einsum(self):

        def _case(f) -> List[Node]:
            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v1 = Tensor(torch.zeros([11, 8, 3, 4]),
                                     mode='partition')
                    self.v2 = Tensor(torch.zeros([11, 9, 4, 5]),
                                     mode='partition')
                    self.t1 = Tensor(torch.zeros(3, 4), mode='replicate')
                    self.t2 = Tensor(torch.zeros(4, 3), mode='replicate')

                def forward(self):
                    f(self.v1, self.v2, self.t1, self.t2)

            m = M()
            graph = EasierTracer().trace(m)

            _fully_load_data_backend_none([m], 'cpu')
            engine = JitEngine(m, graph)
            engine.forward()

            # nodes: v1, v2, t1, t2, ..., out
            return list(graph.nodes)[4:-1]

        f1, = _case(lambda v, w, t, s: torch.einsum('bxjk,bykl->bxjyl', v, w))
        equation, arg1, arg2 = cast(List[Node], f1.args)
        _assert_distributed(f1, 11)

        f2, = _case(lambda v, w, t, s: torch.einsum('bxjk,jk->bx', v, t))
        equation, arg1, arg2 = cast(List[Node], f2.args)
        _assert_distributed(f2, 11)

        f3, = _case(lambda v, w, t, s: torch.einsum('ij,ji->', t, s))
        equation, arg1, arg2 = cast(List[Node], f3.args)
        _assert_replica(f3)


class TestViewSrc:
    pass

"""
TODO
1. multi-res op: a) return input mem; b) two items are actually the same mem
2. memory reused
"""
