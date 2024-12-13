# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Callable, Iterable, List, Optional, Tuple, Type, cast
import pytest
from unittest.mock import MagicMock, Mock, patch

import torch
from torch.fx.node import Node
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.proxy import Proxy

import easier
from easier.core.jit import EasierTracer
from easier.core.passes.layout_alignment.layout_info import \
    PermuteLayout, get_node_layout, is_codegen_node, set_codegen_io_layout
from easier.core.passes.layout_alignment.layout_alignment import \
    PermuteLayoutRewriterBase, align_layout, \
    propagate_layout_info, permute_layout_rewriter_registry

from easier.core.passes.metadata_propagation.metadata import \
    FLOAT32, EasierTensorMeta, Role, StructuredTensorMeta, \
    get_node_meta, set_node_meta, get_meta_from_ir_literal
from easier.core.passes.metadata_propagation.metadata_propagation import \
    MetadataRuleBase, propagate_metadata, metadata_rule_registry
import easier.core.runtime.dist_env as _RuntimeDistEnv
from easier.core.module import \
    EdgeTensor, Gather, VertexSet, VertexTensor, Tensor as _EsrTensor, Scatter
import easier.core.runtime.modules as _Runtime
from easier.core.utils import FX
from tests.utils import torchrun_singlenode


@pytest.fixture(scope='function')
def mock_dist_env():
    # All parameters to activate this fixture
    # must use the same name as this function.
    mock_dist_env = Mock(spec=_RuntimeDistEnv.DistEnv)
    mock_dist_env.world_size = 2    # by default world_size=2
    mock_dist_env.host_rank = 0
    mock_dist_env.comm_device = torch.device('cpu')
    _RuntimeDistEnv._unsafe_worker_local_dist_env = mock_dist_env
    yield mock_dist_env
    _RuntimeDistEnv._unsafe_worker_local_dist_env = None


def _assert_nodes_and_complements(all_nodes: Iterable[Node],
                                  asserting_nodes: List[Node],
                                  assertion: Callable[[Node], bool]):
    negative_asserting_nodes = set(all_nodes) - set(asserting_nodes)

    for node in asserting_nodes:
        assert assertion(node)
    for neg_assert_node in negative_asserting_nodes:
        if not is_codegen_node(neg_assert_node):
            assert not assertion(neg_assert_node)


def assert_having_layout_info(all_nodes: Iterable[Node],
                              asserting_nodes: List[Node]):
    _assert_nodes_and_complements(
        all_nodes, asserting_nodes,
        lambda node: get_node_layout(node) is not None)


class _GraphModule1(GraphModule):
    out_meta: StructuredTensorMeta

    def __new__(cls: Type[GraphModule], *args, **kwargs):
        # The class GraphModule itself, create an anonymous _Impl(GraphModule)
        # every time it's constructed.
        obj = super().__new__(cls, *args, **kwargs)
        metadata_rule_registry[type(obj)] = _GraphModule1_SkipMetaRule
        return obj


def gm(role: Role, x_shp: Tuple[int, ...],
       y_shp: Optional[Tuple[int, ...]] = None):
    # y_shp if specified, should be broadcastable to x_shp
    g = Graph()
    x = g.placeholder('x')
    y = g.placeholder('y')
    add = g.call_function(torch.add, (x, y))
    sum = g.call_method('sum', (add, 1))
    out = g.output([sum])

    # It's ok to not construct concrete EasierVertexMeta etc.
    set_node_meta(x, EasierTensorMeta(shape=x_shp, dtype=FLOAT32, role=role))
    set_node_meta(y, EasierTensorMeta(shape=y_shp or x_shp,
                                      dtype=FLOAT32, role=role))
    set_node_meta(add, EasierTensorMeta(shape=x_shp, dtype=FLOAT32, role=role))
    sum_shp = x_shp[:1] + x_shp[2:]
    set_node_meta(sum, EasierTensorMeta(shape=sum_shp,
                                        dtype=FLOAT32, role=role))
    set_node_meta(out,
                  [EasierTensorMeta(shape=sum_shp, dtype=FLOAT32, role=role)])

    set_codegen_io_layout(x, True)
    set_codegen_io_layout(y, True)
    set_codegen_io_layout(out, [True])

    gm = _GraphModule1({}, g)
    gm.out_meta = get_node_meta(out)
    return gm


class _GraphModule1_SkipMetaRule(MetadataRuleBase):
    def propagate(self, *args, **kwargs):
        gm: _GraphModule1 = self.callee
        return gm.out_meta


def _insert_call_module(x: torch.Tensor, *args, module_name='gm'
                        ) -> torch.Tensor:
    tracer: EasierTracer = x.tracer  # type: ignore
    call_gm_proxy = tracer.create_proxy(
        FX.CALL_MODULE, module_name, (x,) + args, {})
    item0_proxy = tracer.create_proxy(
        FX.CALL_FUNCTION, operator.getitem, (call_gm_proxy, 0), {})
    return item0_proxy  # type: ignore


def test_permute_reduction(mock_dist_env):
    vset = VertexSet(10)
    nv = 5  # partitioned nv

    class PartitionedM(easier.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = VertexTensor(torch.zeros(vset.nv, 2, 3), vset)
            self.v.data = self.v.data[:nv]  # type: ignore

            self.v2 = VertexTensor(torch.zeros(vset.nv, 2, 3), vset)
            self.v2.data = self.v2.data[:nv]  # type: ignore

            self.t = _EsrTensor(torch.zeros(1, 3))  # same ndim as v.sum(1)
            self.gm = gm(Role.VERTEX, (nv, 3), (nv, 3))

        def forward(self):
            # The original forward is:
            # v1 = torch.sum(self.v, dim=1)
            # t1 = self.t / 2
            # gm1 = self.gm(v1, t1)
            # v2 = v1 * gm1
            # t2 = t1 ** 3
            # s = esr.sum(self.v)
            # a2 = esr.mean(self.v2)
            # return

            # The distributed and yet-to-normalize-replica-shape version:
            v1 = torch.sum(self.v, dim=1)  # (nv, 3)
            t1 = self.t / 2

            gm1 = _insert_call_module(v1, t1)  # (nv,)
            gm2 = gm1[:, None]  # (nv, 1)

            v2 = v1 * gm2
            t2 = t1 ** 3

            worker_s = easier.sum(self.v)
            gathered = _Runtime.all_gather_into_tensor(worker_s)
            orig_s = torch.sum(gathered, dim=0, keepdim=True)

            worker_a = easier.mean(self.v2)
            gathered_a = _Runtime.all_gather_into_tensor(worker_a)
            orig_a = torch.mean(gathered_a, dim=0, keepdim=True)

            return

    m = PartitionedM()
    g = EasierTracer(autowrap_modules=[_Runtime]).trace(m)  # type: ignore
    ms, gs = [m], [g]
    ms, gs = propagate_metadata(ms, gs)  # type: ignore

    def _hook_propagate_layout(modules, graphs):
        prop_res = propagate_layout_info(modules, graphs)

        # validate propagation result
        g_prop, = graphs
        attr_v, v1_sum, attr_t, t1_div, \
            call_gm, getitem0, gm2, v2_mul, t2_pow, \
            worker_s, all_gather_s, orig_s, \
            attr_v2, worker_a, all_gather_a, orig_a, \
            ret = g_prop.nodes
        # v2 and worker_a etc. have no layout specified
        assert [
            torch.sum, operator.truediv, operator.mul, operator.pow,
            easier.sum, _Runtime.all_gather_into_tensor, torch.sum,
            easier.mean, _Runtime.all_gather_into_tensor, torch.mean,
        ] == [n.target for n in [
            v1_sum, t1_div, v2_mul, t2_pow,
            worker_s, all_gather_s, orig_s,
            worker_a, all_gather_a, orig_a
        ]]

        assert_having_layout_info(g_prop.nodes, asserting_nodes=[
            attr_v, v1_sum, gm2, v2_mul, worker_s
        ])

        return prop_res

    with patch(
        f'{align_layout.__module__}.{propagate_layout_info.__name__}',
        wraps=_hook_propagate_layout
    ) as mock_pli:
        ms, gs = align_layout(ms, gs)
        mock_pli.assert_called_once()

    (m,), (g_align,) = ms, gs  # type: ignore
    attr_v, v1_sum, attr_t, t1_div, \
        call_gm, getitem0, gm2, v2_mul, t2_pow, \
        worker_s, depermute_worker_s, contig_deperm_s, all_gather_s, orig_s, \
        attr_v2, worker_a, all_gather_a, orig_a, \
        ret = g_align.nodes

    assert m.get_parameter(attr_v.target).shape == (2, 3, nv)

    assert v1_sum.kwargs['dim'] == 0

    assert call_gm.args == (v1_sum, t1_div)
    assert getitem0.args == (call_gm, 0)

    assert worker_s.target == easier.sum
    assert worker_s.kwargs['permuted'] == True

    assert depermute_worker_s.target == torch.permute
    assert depermute_worker_s.args == (worker_s, (2, 0, 1,))

    assert contig_deperm_s.target == 'contiguous'
    assert contig_deperm_s.args == (depermute_worker_s,)

    assert all_gather_s.target == _Runtime.all_gather_into_tensor
    assert all_gather_s.args == (contig_deperm_s,)

    assert worker_a.target == easier.mean
    assert 'permuted' not in worker_a.kwargs  # not marked as permuted

    assert all_gather_a.target == _Runtime.all_gather_into_tensor
    assert all_gather_a.args == (worker_a,)


def test_rewrite_bop_new_node(mock_dist_env):
    """
    Test when during rewriting a batched operation Node is totally erased
    and replaced.
    In such a case new Node is not supposed to have associated metadata or
    layout info.
    """
    vset = VertexSet(10)
    nv = 5  # partitioned nv

    def _adhoc_op(t):
        return t

    def _adhoc_op_rewritten(t):
        return t * 2

    class _AdHocOpMetaRule(MetadataRuleBase):
        def propagate(self, t):
            return get_meta_from_ir_literal(t)
    metadata_rule_registry[_adhoc_op] = _AdHocOpMetaRule

    class _AdHocOpRewriter(PermuteLayoutRewriterBase):
        def rewrite(self, t) -> None:
            with self.graph.inserting_before(self.node.next):
                op2 = self.graph.call_function(_adhoc_op_rewritten, (t,))
            self.node.replace_all_uses_with(op2)
            self.graph.erase_node(self.node)
    permute_layout_rewriter_registry[_adhoc_op] = _AdHocOpRewriter

    def _insert_call_adhoc_op(x: torch.Tensor) -> torch.Tensor:
        tracer: EasierTracer = x.tracer  # type: ignore
        op_proxy = tracer.create_proxy(
            FX.CALL_FUNCTION, _adhoc_op, (x,), {})
        return op_proxy  # type: ignore

    class PartitionedM(easier.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = VertexTensor(torch.zeros(vset.nv, 2, 3), vset)
            self.v.data = self.v.data[:nv]  # type: ignore

            self.t = _EsrTensor(torch.zeros(1, 3))  # same ndim as v.sum(1)
            self.gm = gm(Role.VERTEX, (nv, 3), (nv, 3))

        def forward(self):
            # The distributed and yet-to-normalize-replica-shape version:
            v1 = _insert_call_adhoc_op(self.v)
            t1 = _insert_call_adhoc_op(self.t)

            gm1 = _insert_call_module(v1, t1)  # (nv,)
            gm2 = gm1[:, None, None]

            v2 = v1 * gm2
            t2 = t1 ** 3

            worker_s = easier.sum(self.v)
            gathered = _Runtime.all_gather_into_tensor(worker_s)
            orig_s = torch.sum(gathered, dim=0, keepdim=True)

            return

    m = PartitionedM()
    g = EasierTracer(
        autowrap_modules=[_Runtime],  # type: ignore
        autowrap_functions=[_adhoc_op]  # type: ignore
    ).trace(m)
    ms, gs = [m], [g]
    ms, gs = propagate_metadata(ms, gs)  # type: ignore
    ms, gs = align_layout(ms, gs)

    (m,), (g_align,) = ms, gs  # type: ignore
    attr_v, v1_adhoc, \
        attr_t, t1_adhoc, \
        call_gm, getitem0, gm2, \
        v2_mul, t2_pow, \
        worker_s, depermute_worker_s, contig_deperm, \
        all_gather, orig_s, \
        ret = g_align.nodes

    assert m.get_parameter(attr_v.target).shape == (2, 3, nv)

    assert v1_adhoc.target == _adhoc_op_rewritten
    assert t1_adhoc.target == _adhoc_op

    assert call_gm.args == (v1_adhoc, t1_adhoc)

    assert worker_s.target == easier.sum
    assert worker_s.kwargs['permuted'] == True

    assert depermute_worker_s.target == torch.permute
    assert depermute_worker_s.args == (worker_s, (2, 0, 1,))

    assert contig_deperm.target == 'contiguous'
    assert contig_deperm.args == (depermute_worker_s,)


def test_permute_reduction_keepdim(mock_dist_env):
    # imagine an unbalanced partition to bypass VTensor ctor
    vset = VertexSet(10)
    nv = 5

    class PartitionedM(easier.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = VertexTensor(torch.zeros(vset.nv, 2, 3), vset)
            self.v.data = self.v.data[:nv]  # type: ignore

            self.t = _EsrTensor(torch.zeros(2, 3))
            self.gm = gm(Role.VERTEX, (nv, 2, 3), (nv, 1, 3))

        def forward(self):
            # The original forward is:
            # s = torch.sum(self.v, dim=1)
            # s2 = easier.sum(s)
            # s3 = s2 * 3
            # gm1 = self.gm(self.v, s3)
            # self.t[:,:] = s3
            # return

            # The distributed and yet-to-normalize-replica-shape version:
            s = torch.sum(self.v, dim=1)
            worker_s2 = easier.sum(s)
            gathered = _Runtime.all_gather_into_tensor(worker_s2)
            orig_s2 = torch.sum(gathered, dim=0, keepdim=True)

            s3 = orig_s2 * 3
            gm1 = _insert_call_module(self.v, s3)

            self.t[:, :3] = s3

            return

    m = PartitionedM()
    g = EasierTracer(autowrap_modules=[_Runtime]).trace(m)  # type: ignore
    ms, gs = [m], [g]
    ms, gs = propagate_metadata(ms, gs)  # type: ignore

    def _hook_propagate_layout(modules, graphs):
        prop_res = propagate_layout_info(modules, graphs)

        # validate propagation result
        g_prop, = graphs
        attr_v, s_sum, worker_s2, all_gather, orig_s2, \
            mul_s3, call_gm, getitem0, \
            attr_t, setitem, ret = g_prop.nodes

        assert_having_layout_info(g_prop.nodes, asserting_nodes=[
            attr_v, s_sum, worker_s2
        ])

        return prop_res

    with patch(
        f'{align_layout.__module__}.{propagate_layout_info.__name__}',
        wraps=_hook_propagate_layout
    ) as mock_pli:
        ms, gs = align_layout(ms, gs)
        mock_pli.assert_called_once()

    (m,), (g_align,) = ms, gs

    attr_v, s_sum, \
        worker_s2, de_permute_worker_s2, contig_deperm, all_gather, orig_s2, \
        mul_s3, call_gm, getitem0, \
        attr_t, setitem, ret = g_align.nodes

    # Nodes no longer have valid metadata.

    assert m.get_parameter(attr_v.target).shape == (2, 3, nv)  # permuted

    assert s_sum.kwargs['dim'] == 0

    assert worker_s2.target == easier.sum
    assert worker_s2.kwargs['permuted'] == True

    assert de_permute_worker_s2.target == torch.permute  # reversed
    assert de_permute_worker_s2.args == (worker_s2, (1, 0))

    assert contig_deperm.target == 'contiguous'
    assert contig_deperm.args == (de_permute_worker_s2,)

    assert all_gather.args == (contig_deperm,)
    assert orig_s2.kwargs['dim'] == 0  # already been de-permuted, originally 0

    assert mul_s3.args == (orig_s2, 3)

    assert call_gm.args == (attr_v, mul_s3)

    assert setitem.args == (attr_t, (
        slice(None, None, None),
        slice(None, 3, None),
    ), mul_s3)  # setitem(t, (:, :3), s3) is not changed


def test_easier_primitives(mock_dist_env):
    # Imagine an extremely simple workload that there is no cross-worker
    # communication at all.
    mock_dist_env.rank = 0
    mock_dist_env.batch_isend_irecv.return_value = []
    vset = VertexSet(10)
    vset.vparts = [  # type: ignore
        torch.arange(5),
        torch.arange(5, 10),
    ]
    nv = 5
    idx_part = torch.arange(5)  # local vids
    t0 = torch.zeros(0)

    class PartitionedM(easier.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = VertexTensor(torch.zeros(vset.nv, 2, 3), vset)
            self.v.data = self.v.data[:nv]  # type: ignore

            self.sc = Scatter(idx_part, vset)
            self.sc.nv = idx_part.shape[0]

            self.e = EdgeTensor(torch.zeros(idx_part.shape[0], 7, 8))
            self.e.easier_scatter = self.sc

            self.g = Gather(idx_part, vset)
            self.g.easier_scatter = self.sc

            self.halo = _Runtime.HaloExchanger(
                gather_chunk_idxes=[idx_part, t0],  # type: ignore
                gather_local_vidxes=[idx_part, t0],  # type: ignore
                element_tensor_shape=(2, 3),
                dtype=self.v.dtype,
            )

            self.t = _EsrTensor(torch.zeros(7, 1))

            self.gm = gm(Role.VERTEX, (nv, 2, 3), (nv, 2, 3))

        def forward(self):
            _ = _insert_call_module(self.v, self.v)

            v2 = self.v * 2
            chunk = self.halo(v2)
            e = self.g(chunk)  # (ne, 2, 3)
            e2 = e[:, :1]  # (ne, 1, 3)
            e3 = e2[:, :1, 0]  # (ne, 1)
            self.e[:, 1, :8] = e3

            self.e[:, :7] = self.t
            self.e[:, :7, :8] = self.t[None, None, None]

            v3 = self.sc(e2)
            v4 = self.sc(self.e)

            return

    m = PartitionedM()
    tracer = EasierTracer()  # type: ignore
    _orig_is_leaf_module = tracer.is_leaf_module

    def _is_leaf_module(this: EasierTracer, m: torch.nn.Module, name: str):
        return name == "halo" or _orig_is_leaf_module(m, name)
    tracer.is_leaf_module = _is_leaf_module.__get__(tracer)
    g = tracer.trace(m)

    ms, gs = [m], [g]
    ms, gs = propagate_metadata(ms, gs)  # type: ignore

    def _hook_propagate_layout(modules, graphs):
        prop_res = propagate_layout_info(modules, graphs)

        # validate propagation result
        g_prop, = graphs
        attr_v, _callgm, _getitem0, v2_mul, halo, \
            e_gather, e2_getitem, e3_getitem, \
            attr_e, e_setitem_e3, \
            attr_t, e_setitem_t, \
            leftpad_t, e_setitem2_t, \
            v3_sc, v4_sc, \
            ret = g_prop.nodes

        assert_having_layout_info(g_prop.nodes, asserting_nodes=[
            attr_v, v2_mul, halo, e_gather, e2_getitem, e3_getitem,
            attr_e, e_setitem_e3, e_setitem_t, e_setitem2_t,
            v3_sc, v4_sc
        ])

        return prop_res

    with patch(
        f'{align_layout.__module__}.{propagate_layout_info.__name__}',
        wraps=_hook_propagate_layout
    ) as mock_pli:
        ms, gs = align_layout(ms, gs)
        mock_pli.assert_called_once()

    (m,), (g_align,) = ms, gs
    m = cast(PartitionedM, m)

    attr_v, _gm, getitem0, v2_mul, halo, e_gather, e2_getitem, e3_getitem, \
        attr_e, e_setitem_e3, \
        attr_t, bc_shpnorm_t, perm_shpnorm_t, e_setitem_t, \
        leftpad_t, bc_shpnorm2_t, perm_shpnorm2_t, e_setitem2_t, \
        v3_sc, v4_sc, \
        ret = g_align.nodes

    assert halo.op == FX.CALL_MODULE and halo.target == 'halo'
    assert halo.args == (v2_mul,)
    assert m.halo.layout_permuted == True
    assert m.halo.chunk_v.shape == (2, 3, 5)

    assert e_gather.target == 'g'
    assert e_gather.args == (halo, True)  # (..., permuted=True)

    assert v3_sc.target == 'sc'
    assert v3_sc.args == (e2_getitem, True)  # (..., permuted=True)

    assert e2_getitem.target == operator.getitem
    assert e2_getitem.args == (e_gather, (slice(1), slice(None), slice(None)))

    assert e3_getitem.target == operator.getitem
    assert e3_getitem.args == (e2_getitem, (slice(1), 0, slice(None)))

    assert e_setitem_e3.target == operator.setitem
    assert e_setitem_e3.args == (attr_e, (1, slice(8), slice(None)),
                                 e3_getitem)

    # self.t is a replica, and the `setitem` has broadcasting
    assert bc_shpnorm_t.target == operator.getitem
    assert bc_shpnorm_t.args == (attr_t, (None,))
    assert perm_shpnorm_t.target == torch.permute
    assert perm_shpnorm_t.args == (bc_shpnorm_t, (1, 2, 0))
    assert e_setitem_t.target == operator.setitem
    assert e_setitem_t.args == (attr_e, (slice(7), slice(None), slice(None)),
                                perm_shpnorm_t)

    assert leftpad_t.target == operator.getitem
    assert leftpad_t.args == (attr_t, (None, None, None))
    # padded t: (1,1,1,7,1)
    assert bc_shpnorm2_t.target == "view"
    assert bc_shpnorm2_t.args == (leftpad_t, (1, 7, 1))
    assert perm_shpnorm2_t.target == torch.permute
    assert perm_shpnorm2_t.args == (bc_shpnorm2_t, (1, 2, 0))
    assert e_setitem2_t.target == operator.setitem
    assert e_setitem2_t.args == (attr_e, (slice(7), slice(8), slice(None)),
                                 perm_shpnorm2_t)


def test_permute__dim_for_output(mock_dist_env):
    # some ops `dim` parameter is regarding the expected result tensor,
    # not regarding an input, e.g.
    # unsqueeze(rand(nv,2,3), dim=3).shape == (nv,2,3,1) # or dim=-1

    vset = VertexSet(10)
    nv = 5

    class PartitionedM(easier.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = VertexTensor(torch.zeros(vset.nv, 2, 3), vset)
            self.v.data = self.v.data[:nv]  # type: ignore

            self.gm = gm(Role.VERTEX, (nv, 2, 3), (nv, 2, 3))

        def forward(self):
            _ = _insert_call_module(self.v, self.v)
            v1 = torch.unsqueeze(self.v, 1)
            v2 = torch.unsqueeze(self.v, -1)
            v3 = self.v.repeat(1, 5, 6)

            stk = torch.stack(  # (nv, 1)
                [v1[:, 0, 1, 2], v2[:, 1, 2, 0], v3[:, 3, 4]], dim=-1)

            return

    m = PartitionedM()
    g = EasierTracer(autowrap_modules=[_Runtime]).trace(m)  # type: ignore
    ms, gs = [m], [g]
    ms, gs = propagate_metadata(ms, gs)  # type: ignore
    ms, gs = align_layout(ms, gs)
    g_align, = gs

    attr_v, _callgm, _getitem0, \
        v1, v2, v3, slc1, slc2, slc3, stk, ret = g_align.nodes
    assert v1.kwargs['dim'] == 0
    assert v2.kwargs['dim'] == 2
    assert v3.args == (attr_v, (5, 6, 1))

    colon = slice(None)
    assert slc1.args == (v1, (0, 1, 2, colon))
    assert slc2.args == (v2, (1, 2, 0, colon))
    assert slc3.args == (v3, (3, 4, colon))

    assert stk.kwargs['dim'] == 0


def test_cross_graph(mock_dist_env):
    vset = VertexSet(10)
    nv = 5

    v = VertexTensor(torch.zeros(vset.nv, 2, 3), vset)
    v.data = v.data[:nv]  # type: ignore

    class PM1(easier.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = v

            self.gm = gm(Role.VERTEX, (nv, 2, 3), (nv, 2, 3))

        def forward(self):
            _insert_call_module(self.v, self.v)
            return

    class PM2(easier.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = v

        def forward(self):
            # original:
            v2 = self.v.sum(dim=2)
            worker_s = easier.sum(v2)
            gathered = _Runtime.all_gather_into_tensor(worker_s)
            orig_s = torch.sum(gathered, dim=0, keepdim=True)
            return

    m1, m2 = PM1(), PM2()
    tracer = EasierTracer(autowrap_modules=[_Runtime])  # type: ignore
    g1, g2 = tracer.trace(m1), tracer.trace(m2)

    ms, gs = [m1, m2], [g1, g2]
    ms, gs = propagate_metadata(ms, gs)  # type: ignore
    ms, gs = align_layout(ms, gs)
    ga1, ga2 = gs

    attr_v, v2, worker_sum, deperm, contig, allgather, orig_sum, out = ga2.nodes

    assert v2.target == 'sum'
    assert v2.kwargs['dim'] == 1

    assert worker_sum.target == easier.sum
    assert worker_sum.kwargs['permuted'] == True

    assert deperm.target == torch.permute
    assert deperm.args == (worker_sum, (1, 0))


def worker__test_sync_remember_layout(
        local_rank: int, world_size: int,
):
    nv = 50
    ne = 30
    vset = VertexSet(nv)

    idx1 = torch.arange(ne)
    idx2 = idx1.flip(dims=[0])
    raw_v = torch.arange(nv * 2 * 3 * 4).reshape(nv, 2, 3, 4)
    raw_e = torch.arange(ne * 5 * 6).reshape(ne, 5, 6)

    class M(easier.Module):
        def __init__(self) -> None:
            super().__init__()
            self.v = VertexTensor(raw_v, vset)
            self.s = Scatter(idx1, vset)
            self.e = EdgeTensor(raw_e)
            self.e.easier_scatter = self.s
            self.g = Gather(idx2, vset)
            self.g.easier_scatter = self.s

            # batchsizes are dynamically partitions of nv and ne...
            # But it's ok since we don't run it.
            self.gm = gm(Role.VERTEX, (nv, 2, 3, 4))
            self.gm2 = gm(Role.EDGE, (ne, 5, 6))

        def forward(self):
            v2 = _insert_call_module(self.v, self.v)
            e2 = _insert_call_module(self.e, self.e, module_name='gm2')
            _ = self.s(self.g(self.v))
            return

    m = M()

    # Simplified jit()
    import easier.core.passes as passes
    from torch.fx.graph_module import GraphModule
    _RuntimeDistEnv.init_dist_env('cpu')
    modules: List[torch.nn.Module] = [m]
    graphs = [EasierTracer().trace(m)]
    modules, graphs = passes.propagate_metadata(modules, graphs)
    modules, graphs = passes.distribute_dataflow(modules, graphs, 'torch')
    modules, graphs = passes.propagate_metadata(modules, graphs)
    (m,), (g,) = passes.align_layout(modules, graphs)
    m = cast(M, m)

    # assert parameters really get partitioned and permuted
    nv_part = vset.vparts[local_rank].shape[0]
    assert nv_part < nv
    assert m.v.shape == (2, 3, 4, nv_part)
    v_synced = m.v.sync()
    assert v_synced.shape == (nv, 2, 3, 4)
    assert torch.equal(v_synced, raw_v)

    ne_part = m.s.reordered_sliced_pos.shape[0]
    assert ne_part < ne
    assert m.e.shape == (5, 6, ne_part)
    e_synced = m.e.sync()
    assert e_synced.shape == (ne, 5, 6)
    assert torch.equal(e_synced, raw_e)


def test_tensor_sync_remember_layout():
    torchrun_singlenode(2, worker__test_sync_remember_layout)
