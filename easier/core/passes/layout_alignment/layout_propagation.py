# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum
import itertools
import operator
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union, cast
from typing_extensions import TypeAlias
import more_itertools

import torch
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx.graph_module import GraphModule
from easier.core.passes.layout_alignment.layout_info import \
    PermuteLayout, StructuredLayout, get_codegen_io_layout, \
    is_codegen_node, mark_codegen_node, get_node_layout, set_node_layout

from easier.core.passes.metadata_propagation.metadata import \
    EasierTensorMeta, Role, get_node_meta
from easier.core.utils import FX, isinst_checker, tree_map
import easier.core.module as _EsrMod
import easier.core.runtime.modules as _Runtime


EasierDistTensor: TypeAlias = Union[_EsrMod.VertexTensor, _EsrMod.EdgeTensor]


@dataclass
class _GraphModuleInputOutputLayouts:
    placeholders: List[Node]
    input_layouts: List[PermuteLayout]

    output: Node
    output_layouts: List[PermuteLayout]


class _OpKind(Enum):
    # inputs and outputs are all replicas, fully SPMD.
    LOCAL = 0
    # inputs and outputs are all distributed tensors,
    # also including 'get_attr' that has no input and returns a dist tensor.
    DIST = 1
    # inputs are distributed tensors, outputs are replicas.
    REDUCE = 2


class CrossGraphLayoutPropagator:
    def __init__(
            self,
            disttensor2getattrs: Dict[EasierDistTensor, Set[Node]]
    ) -> None:
        self.disttensor_to_getattrs = disttensor2getattrs

        getattr_to_disttensor: Dict[Node, EasierDistTensor] = {}
        for p, getattrs in disttensor2getattrs.items():
            for get_attr_node in getattrs:
                getattr_to_disttensor[get_attr_node] = p

        self.getattr_to_disttensor = getattr_to_disttensor

    def try_propagate_along_same_disttensor(
            self, node: Node, layout: PermuteLayout
    ):
        dt = self.getattr_to_disttensor.get(node, None)
        if dt is not None:
            same_dt_getattrs = self.disttensor_to_getattrs[dt]
            for other_getattr in same_dt_getattrs:
                if other_getattr is not node:
                    self.propagate_distributed_op(other_getattr, layout)

    def terminate_on_local_reduction(
            self, node: Node, layout: PermuteLayout
    ):
        # When this `node` is a reduction.
        # Its result, although is a replica, inherits the layout of this op.
        users = list(node.users)
        assert len(users) == 1 and \
            users[0].target == _Runtime.all_gather_into_tensor
        set_node_layout(node, layout)

    def get_distributed_op_kind(self, node: Node) -> _OpKind:
        input_roles: Set[Role] = set()
        output_roles: Set[Role] = set()

        def _role_collecter(roles: Set[Role]):
            def _role_collect(x):
                assert isinstance(x, EasierTensorMeta)
                roles.add(x.role)
                return None
            return _role_collect

        imetas = [get_node_meta(arg) for arg in node.all_input_nodes]
        _ = tree_map(imetas, _role_collecter(input_roles))
        input_dist_roles = input_roles - set([Role.REPLICA])

        nmeta = get_node_meta(node)
        _ = tree_map(nmeta, _role_collecter(output_roles))
        assert len(output_roles) == 1, \
            "Operations remaining in the graph and having layout info" \
            " must not return tensors with more than one Role"
        output_role, = output_roles

        if len(input_dist_roles) == 1:
            if output_role.is_distributed:
                return _OpKind.DIST
            else:
                assert node.target in [
                    _EsrMod.sum, _EsrMod.norm, _EsrMod.mean
                ]
                return _OpKind.REDUCE

        else:
            if output_role.is_distributed:
                assert node.op == FX.GET_ATTR, \
                    "Distributed-op-propagation reaches a Node" \
                    f" {node.format_node()} without an input distributed" \
                    " tensor but returning an distributed tensor"
                return _OpKind.DIST
            else:
                return _OpKind.LOCAL

    def propagate_distributed_op(self, node: Node, layout: PermuteLayout):
        if is_codegen_node(node):
            # Node[op=call_module] to multiple-output GraphModules are included
            # in this set and excluded.
            return

        prev_layout = get_node_layout(node)
        if prev_layout is not None:
            if prev_layout != layout:
                raise NotImplementedError(
                    "Conflict layout detected, we may need merging policy")
            return

        dist_op_kind = self.get_distributed_op_kind(node)
        if dist_op_kind == _OpKind.DIST:

            set_node_layout(node, layout)

            for arg in node.all_input_nodes:
                self.propagate_distributed_op(arg, layout)
            for user in node.users:
                self.propagate_distributed_op(user, layout)

            self.try_propagate_along_same_disttensor(node, layout)

        elif dist_op_kind == _OpKind.REDUCE:
            # NOTE the propagation won't reach a worker-local reduction
            # as one of its users, because there are always a `all_gather` and
            # a replica reduction (replica-in, replica-out, fully SPMD)
            # after the worker-local reduction.
            self.terminate_on_local_reduction(node, layout)

    def start_from_codegen_nodes(
        self,
            starting_codegen_arg_nodes: List[Tuple[Node, PermuteLayout]],
            starting_codegen_user_nodes: List[Tuple[Node, PermuteLayout]]
    ):
        for arg, layout in starting_codegen_arg_nodes:
            self.propagate_distributed_op(arg, layout)

        for user, layout in starting_codegen_user_nodes:
            self.propagate_distributed_op(user, layout)


def get_graph_module_input_output_layouts(gm_graph: Graph):
    def _assert_placeholder_layout_specified(ph: Node) -> PermuteLayout:
        ph_cg_layout = get_codegen_io_layout(ph)
        assert isinstance(ph_cg_layout, bool), \
            "Codegen unit must explicit specify the placeholder layout"
        return PermuteLayout.TRUE if ph_cg_layout else PermuteLayout.FALSE

    def _assert_output_layouts_specified(output: Node) -> List[PermuteLayout]:
        output_cg_layouts = get_codegen_io_layout(output)
        assert (
            isinstance(output_cg_layouts, list) and
            all(isinstance(ol, bool) for ol in output_cg_layouts)
        ), "Codegen unit must explicit specify the output layouts"

        output_layouts: List[PermuteLayout] = list(
            PermuteLayout.TRUE if cg_layout else PermuteLayout.FALSE
            for cg_layout in output_cg_layouts
        )
        return output_layouts

    placeholders: List[Node] = []
    input_layouts: List[PermuteLayout] = []

    for ph_node in gm_graph.nodes:
        ph_node: Node
        if ph_node.op == FX.PLACEHOLDER:
            layout = _assert_placeholder_layout_specified(ph_node)
            placeholders.append(ph_node)
            input_layouts.append(layout)

    output: Node = next(iter(reversed(gm_graph.nodes)))  # type: ignore
    output_layouts = _assert_output_layouts_specified(output)

    return _GraphModuleInputOutputLayouts(
        placeholders=placeholders, input_layouts=input_layouts,
        output=output, output_layouts=output_layouts
    )


@dataclass
class CollectCodegenNodesResult:
    """
    Any collected Node-Layout pair (i.e. an edge of the dataflow graph)
    will only represented a single tensor (multiple pairs may refer to the
    same tensor) rather than a structure (e.g. list of tensors).
    """
    starting_codegen_arg_nodes: List[Tuple[Node, PermuteLayout]]
    starting_codegen_user_nodes: List[Tuple[Node, PermuteLayout]]


def collect_and_mark_codegen_nodes(
        modules: List[torch.nn.Module], graphs: List[Graph]
) -> CollectCodegenNodesResult:
    """
    Side effects:
    -   The `op=call_module` Nodes to GraphModules and their following
        tuple-unpacking `target=getitem` Nodes will be marked as
        _codegen nodes_ in their `node.meta:dict`.
    """

    def _set_new_layout(node: Node, new_layout: PermuteLayout):
        # Set new layout with policies.
        # We don't regardlessly overwrite layout.
        prev_layout = get_node_layout(node)
        if prev_layout is not None:
            if prev_layout != new_layout:
                raise NotImplementedError(
                    "Conflict layout detected, we may need merging policy")
        else:
            set_node_layout(node, new_layout)

    # Different Node[op=call_module] may take the same input Node
    starting_arg_nodes: List[Tuple[Node, PermuteLayout]] = []
    starting_user_nodes: List[Tuple[Node, PermuteLayout]] = []

    for m, g in zip(modules, graphs):
        for call_gm in g.nodes:
            call_gm: Node

            if call_gm.op == FX.CALL_MODULE:
                submod_path: str = call_gm.target  # type: ignore
                gm = m.get_submodule(submod_path)
                if isinstance(gm, GraphModule):

                    mark_codegen_node(call_gm)

                    gm_inout_layouts = \
                        get_graph_module_input_output_layouts(gm.graph)
                    assert len(call_gm.kwargs) == 0, \
                        "Node calling a GraphModule should not have kwargs"

                    for arg_i, arg in enumerate(call_gm.args):
                        if isinstance(arg, Node):
                            starting_arg_nodes.append(
                                (arg, gm_inout_layouts.input_layouts[arg_i]))

                    output_layouts: List[PermuteLayout] = \
                        gm_inout_layouts.output_layouts

                    # We just propagate layout info on all resultant and
                    # tuple-unpacking `getitem` Nodes. We don't bother storing
                    # a list of layouts on the `Node[call_module]` itself.
                    for out_i, tuple_getitem in enumerate(call_gm.users):
                        assert tuple_getitem.target == operator.getitem, \
                            "Immediate users of Node calling a GraphModule" \
                            " must be operator.getitem Nodes"
                        # Fusion pass copied metadata of terminal Nodes within
                        # the GraphModule to those `tuple_getitem` Nodes
                        # (there are always such `tuple_getitem` Nodes even
                        # the GraphModule only has one output tensor).
                        assert isinstance(
                            get_node_meta(tuple_getitem), EasierTensorMeta
                        ), "no more nested"

                        mark_codegen_node(tuple_getitem)

                        output_layout = output_layouts[out_i]

                        for user in tuple_getitem.users:
                            starting_user_nodes.append((user, output_layout))

                        _set_new_layout(tuple_getitem, output_layout)

    return CollectCodegenNodesResult(starting_arg_nodes, starting_user_nodes)


KEY__GETATTR_MODULENAME = 'easier__layoutAlignment__getAttrClassName'


def propagate_layout_info(
        modules: List[torch.nn.Module], graphs: List[Graph]
) -> Dict[EasierDistTensor, Set[Node]]:

    coll_res = collect_and_mark_codegen_nodes(modules, graphs)

    crossgraph_getattrs: Dict[EasierDistTensor, Set[Node]] = {}
    for m, g in zip(modules, graphs):
        for node in g.nodes:
            node: Node
            if node.op == FX.GET_ATTR:
                # Maybe `_constant_tensor0`, not a parameter or a buffer.
                submod, _, attrname = (cast(str, node.target)).rpartition('.')
                p = getattr(m.get_submodule(submod), attrname)

                if isinstance(p, (_EsrMod.VertexTensor, _EsrMod.EdgeTensor)):
                    crossgraph_getattrs.setdefault(p, set()).add(node)

                    # For debug purpose only
                    node.meta[KEY__GETATTR_MODULENAME] = m.__class__.__name__

    CrossGraphLayoutPropagator(crossgraph_getattrs).start_from_codegen_nodes(
        coll_res.starting_codegen_arg_nodes,
        coll_res.starting_codegen_user_nodes,
    )

    return crossgraph_getattrs
