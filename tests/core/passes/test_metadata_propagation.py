# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import List, cast
import pytest
import torch

from torch.fx.graph import Graph
from torch.fx.node import Node

from easier.core.jit import EasierTracer

from easier.core.module import Selector, Reducer, Tensor
from easier.core.passes.metadata_propagation.metadata import \
    INT32, EasierTensorMeta, EasierTensorMeta, \
    Role, ScalarType
from easier.core.passes.metadata_propagation.utils import \
    Validation as V
from easier.core.passes.utils import \
    FX
from easier.core import passes
import easier as esr


class TestMetadataPropagation:

    def test_easier_primitives(self):
        class M(esr.Module):
            def __init__(self):
                super().__init__()
                self.s = Selector(torch.LongTensor([1, 2, 3, 4]))
                self.r = Reducer(torch.LongTensor([6, 7, 8, 9]), 22)
                self.v = Tensor(torch.zeros([11, 3]), dist='partition')

            def forward(self):
                e = self.s(self.v)
                v2 = self.r(e)

        m = M()
        graph = EasierTracer().trace(m)
        [m], [graph] = passes.propagate_metadata([m], [graph])

        getattr_v, call_s, call_r, output = graph.nodes

        vmeta = cast(EasierTensorMeta, V.assert_non_structured(getattr_v))
        assert vmeta.role == Role.PARTITION
        assert vmeta.shape == (11, 3)

        smeta = cast(EasierTensorMeta, V.assert_non_structured(call_s))
        assert smeta.role == Role.PARTITION
        assert smeta.shape == (4, 3)

        rmeta = cast(EasierTensorMeta, V.assert_non_structured(call_r))
        assert rmeta.role == Role.PARTITION
        assert rmeta.shape == (22, 3)

    def test_getsetitem(self):
        def _case(f) -> List[Node]:
            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v = Tensor(torch.zeros([11, 2, 3, 4]),
                                    dist='partition')
                    self.t = Tensor(torch.zeros([5, 6, 7], dtype=torch.int32),
                                    dist='replicate')

                def forward(self):
                    f(self.v, self.t)

            m = M()
            graph = EasierTracer().trace(m)
            passes.propagate_metadata([m], [graph])
            # nodes: v, t, ..., out
            return list(graph.nodes)[2:-1]

        slice1, = _case(lambda v, t: v[:])
        v1meta = V.must_of(
            V.assert_non_structured(slice1), EasierTensorMeta)
        assert v1meta.shape == (11, 2, 3, 4)

        slice2, = _case(lambda v, t: v[:, None, :, 1, 2:4])
        v2meta = V.must_of(
            V.assert_non_structured(slice2), EasierTensorMeta)
        assert v2meta.shape == (11, 1, 2, 2)

        def _bc_t_113_into_23(v, t):
            t[0, :2, :3] = t[:1, :1, :3]
        gett3, sett3, = _case(_bc_t_113_into_23)
        gett3meta = V.must_of(V.assert_non_structured(gett3), EasierTensorMeta)
        assert gett3meta.shape == (1, 1, 3)
        sett3meta = V.must_of(V.assert_non_structured(sett3), EasierTensorMeta)
        assert sett3meta.shape == (5, 6, 7)

        def _case4(v, t):
            t2 = t + 2  # out-of-index, but ok for metadata
            v[:, None, 0:2, ..., None, t2]
        calct2, slice4, = _case(_case4)
        v4meta = cast(EasierTensorMeta, V.assert_non_structured(slice4))
        assert v4meta.shape == (11, 1, 2, 3, 1, 5, 6, 7)

        def _case5(v, t):
            v[:] = v * 2
        mul, setv, = _case(_case5)
        mulmeta = V.must_of(V.assert_non_structured(mul), EasierTensorMeta)
        assert mulmeta.shape == (11, 2, 3, 4)
        setvmeta = V.must_of(
            V.assert_non_structured(setv), EasierTensorMeta)
        assert setvmeta.shape == (11, 2, 3, 4)


class TestRules:
    def test_arith_ops(self):

        def _case(f) -> List[Node]:
            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v1 = Tensor(torch.zeros([11, 2, 3, 4]),
                                     dist='partition')
                    self.v2 = Tensor(torch.zeros([11, 1, 1, 1]),
                                     dist='partition')

                def forward(self):
                    f(self.v1, self.v2)

            m = M()
            graph = EasierTracer().trace(m)
            passes.propagate_metadata([m], [graph])
            # nodes: v1, v2, ..., out
            return list(graph.nodes)[2:-1]

        add, = _case(torch.add)
        assert V.assert_non_structured(add).shape == (11, 2, 3, 4)

        pow, = _case(torch.pow)
        assert V.assert_non_structured(pow).shape == (11, 2, 3, 4)

    def test_reduction(self):
        def _case(f) -> List[Node]:

            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v = Tensor(torch.zeros(
                        [11, 2, 3, 4], dtype=torch.int32), dist='partition')

                def forward(self):
                    f(self.v)

            m = M()
            graph = EasierTracer().trace(m)
            passes.propagate_metadata([m], [graph])
            # nodes: v, ..., out
            return list(graph.nodes)[1:-1]

        sum_full, sum_0rank = _case(lambda x: esr.sum(x).sum())
        meta: EasierTensorMeta = V.assert_non_structured(sum_full)
        assert meta == EasierTensorMeta((1, 2, 3, 4), INT32, Role.REPLICA)
        meta: EasierTensorMeta = V.assert_non_structured(sum_0rank)
        assert meta == EasierTensorMeta((), INT32, Role.REPLICA)

    def test_overloads(self):
        def _case(fv=lambda x: x, ft=lambda x: x) -> List[Node]:

            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v = esr.Tensor(torch.zeros(
                        [11, 2, 3, 4], dtype=torch.int32), dist='partition')
                    self.t = esr.Tensor(torch.zeros(
                        [11, 2, 3, 4], dtype=torch.int32), dist='replicate')

                def forward(self):
                    v = self.v
                    t = self.t
                    fv(v)
                    ft(t)

            m = M()
            graph = EasierTracer().trace(m)
            passes.propagate_metadata([m], [graph])
            # nodes: v, t, ..., out
            return list(graph.nodes)[2:-1]

        # NOTE FX hooks functions when users are accessing a global variable
        # (here the variable is the function name)
        # so we need the lambda expression to ensure such an access pattern,
        # instead of writing `fv=esr.sum` which is merely a function object.
        sum_full, = _case(fv=lambda x: esr.sum(x))
        meta: EasierTensorMeta = V.assert_non_structured(sum_full)
        assert meta == EasierTensorMeta((1, 2, 3, 4), INT32, Role.REPLICA)

        sum_full, = _case(ft=lambda x: torch.sum(x))
        meta: EasierTensorMeta = V.assert_non_structured(sum_full)
        assert meta == EasierTensorMeta((), INT32, Role.REPLICA)

        sum_dims_v, sum_dims_t, = _case(
            fv=lambda x: x.sum([1, 2]),
            ft=lambda x: x.sum([1, 2])
        )
        meta: EasierTensorMeta = V.assert_non_structured(sum_dims_v)
        assert meta == EasierTensorMeta((11, 4), INT32, Role.PARTITION)
        meta: EasierTensorMeta = V.assert_non_structured(sum_dims_t)
        assert meta == EasierTensorMeta((11, 4), INT32, Role.REPLICA)

        sum_dims_v, sum_dims_t, = _case(
            fv=lambda x: x.sum([1, 2], keepdim=True),
            ft=lambda x: x.sum([1, 2], keepdim=True)
        )
        meta: EasierTensorMeta = V.assert_non_structured(sum_dims_v)
        assert meta == EasierTensorMeta((11, 1, 1, 4), INT32, Role.PARTITION)
        meta: EasierTensorMeta = V.assert_non_structured(sum_dims_t)
        assert meta == EasierTensorMeta((11, 1, 1, 4), INT32, Role.REPLICA)

    def test_where(self):

        def _case(f) -> List[Node]:
            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v1 = Tensor(torch.zeros([11, 2, 3, 4]),
                                     dist='partition')
                    self.v2 = Tensor(torch.zeros([11, 1, 1, 1]),
                                     dist='partition')

                def forward(self):
                    f(self.v1 == self.v2, self.v1, self.v2)

            m = M()
            graph = EasierTracer().trace(m)
            passes.propagate_metadata([m], [graph])
            # nodes: v1, v2, ..., out
            return list(graph.nodes)[:-1]

        v1, v2, eq, where = _case(torch.where)
        assert V.assert_non_structured(eq).shape == (11, 2, 3, 4)
        assert V.assert_non_structured(where).shape == (11, 2, 3, 4)
        assert where.op == FX.CALL_FUNCTION and where.target == torch.where
        assert where.args == (eq, v1, v2)

        v1, v2, eq, where = _case(lambda c, x, y: x.where(c, y))
        assert V.assert_non_structured(eq).shape == (11, 2, 3, 4)
        assert V.assert_non_structured(where).shape == (11, 2, 3, 4)
        assert where.op == FX.CALL_METHOD and where.target == 'where'
        assert where.args == (v1, eq, v2)

    def test_matmul(self):

        def _case(f) -> List[Node]:
            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v1 = Tensor(torch.zeros([11, 8, 3, 4]),
                                     dist='partition')
                    self.v2 = Tensor(torch.zeros([11, 1, 4, 5]),
                                     dist='partition')
                    self.t1 = Tensor(torch.zeros(3, 4), dist='replicate')
                    self.t2 = Tensor(torch.zeros(4, 5), dist='replicate')

                def forward(self):
                    f(self.v1, self.v2, self.t1, self.t2)

            m = M()
            graph = EasierTracer().trace(m)
            passes.propagate_metadata([m], [graph])
            # nodes: v1, v2, t1, t2, ..., out
            return list(graph.nodes)[4:-1]

        mm, = _case(lambda v, w, t, s: v @ w)
        arg1, arg2 = cast(List[Node], mm.args)
        assert V.assert_non_structured(arg1).shape == (11, 8, 3, 4)
        assert V.assert_non_structured(arg2).shape == (11, 1, 4, 5)
        assert V.assert_non_structured(mm).shape == (11, 8, 3, 5)
        assert V.assert_non_structured(mm).role == Role.PARTITION

        t0, mm, = _case(lambda v, w, t, s: v @ t[0])
        arg1, arg2 = cast(List[Node], mm.args)
        assert V.assert_non_structured(arg1).shape == (11, 8, 3, 4)
        assert V.assert_non_structured(arg2).shape == (4,)
        assert V.assert_non_structured(mm).shape == (11, 8, 3)
        assert V.assert_non_structured(mm).role == Role.PARTITION

        t0, mm, = _case(lambda v, w, t, s: t[0] @ w)
        arg1, arg2 = cast(List[Node], mm.args)
        assert V.assert_non_structured(arg1).shape == (4,)
        assert V.assert_non_structured(arg2).shape == (11, 1, 4, 5)
        assert V.assert_non_structured(mm).shape == (11, 1, 5)
        assert V.assert_non_structured(mm).role == Role.PARTITION

    def test_einsum(self):

        def _case(f) -> List[Node]:
            class M(esr.Module):
                def __init__(self):
                    super().__init__()
                    self.v1 = Tensor(torch.zeros([11, 8, 3, 4]),
                                     dist='partition')
                    self.v2 = Tensor(torch.zeros([11, 9, 4, 5]),
                                     dist='partition')
                    self.t1 = Tensor(torch.zeros(3, 4), dist='replicate')
                    self.t2 = Tensor(torch.zeros(4, 3), dist='replicate')

                def forward(self):
                    f(self.v1, self.v2, self.t1, self.t2)

            m = M()
            graph = EasierTracer().trace(m)
            passes.propagate_metadata([m], [graph])
            # nodes: v1, v2, t1, t2, ..., out
            return list(graph.nodes)[4:-1]

        f1, = _case(lambda v, w, t, s: torch.einsum('bxjk,bykl->bxjyl', v, w))
        equation, arg1, arg2 = cast(List[Node], f1.args)
        assert V.assert_non_structured(arg1).shape == (11, 8, 3, 4)
        assert V.assert_non_structured(arg2).shape == (11, 9, 4, 5)
        assert V.assert_non_structured(f1).shape == (11, 8, 3, 9, 5)
        assert V.assert_non_structured(f1).role == Role.PARTITION

        f2, = _case(lambda v, w, t, s: torch.einsum('bxjk,jk->bx', v, t))
        equation, arg1, arg2 = cast(List[Node], f2.args)
        assert V.assert_non_structured(arg1).shape == (11, 8, 3, 4)
        assert V.assert_non_structured(arg2).shape == (3, 4)
        assert V.assert_non_structured(f2).shape == (11, 8)
        assert V.assert_non_structured(f2).role == Role.PARTITION

        f3, = _case(lambda v, w, t, s: torch.einsum('ij,ji->', t, s))
        equation, arg1, arg2 = cast(List[Node], f3.args)
        assert V.assert_non_structured(arg1).shape == (3, 4)
        assert V.assert_non_structured(arg2).shape == (4, 3)
        assert V.assert_non_structured(f3).shape == ()
        assert V.assert_non_structured(f3).role == Role.REPLICA
