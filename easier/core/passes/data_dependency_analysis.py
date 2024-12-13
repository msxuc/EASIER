# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
import itertools
import operator
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast, FrozenSet
from typing_extensions import TypeAlias

import torch
import torch.overrides
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument, map_arg
from easier.core.passes.metadata_propagation.metadata import \
    EasierTensorMeta, Role, View, ViewType, get_node_meta
from easier.core.passes.metadata_propagation.utils import \
    Validation as MetaV

from easier.core.utils import logger, EasierJitException
import easier.core.module as esr

from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, tree_map

# `ViewSrc` are Nodes that:
# - represent individual storage of the tensors.
# - are the torch operations that allocate the memory blocks for the tensors;
#
# The `metadata.View.src` carries the storage identities.
#
# However, metadata propagation leaves the view sources to be None for the
# `type==ALLOCATED` cases (but the source Node will be very obvious though),
# we need to pick up the associated Nodes first.
ViewSrc: TypeAlias = Union[
    Node,               # non-multi-result Nodes, including get_attr
    Tuple[Node, int],   # item of multi-result Nodes
]

KEY__DATA_DEPENDENCY_INPUTS = 'easier_dataDependency_inputs'
KEY__DATA_DEPENDENCY_USERS = 'easier_dataDependency_users'


def get_data_dependency_inputs(node: Node) -> List[Node]:
    # Return a clone to avoid manipulating the node meta dict.
    return list(node.meta.get(KEY__DATA_DEPENDENCY_INPUTS, []))


def get_data_dependency_users(node: Node) -> List[Node]:
    # Return a clone to avoid manipulating the node meta dict.
    return list(node.meta.get(KEY__DATA_DEPENDENCY_USERS, []))


class DataDependencyAnalyzer(EasierInterpreter):
    """
    Some Nodes (Node-plus-index for multi-result operators like `linalg.svd`)
    are themselves representing, or 1:1-corresponding to,
    the view sources (e.g. the underlying memory blocks),
    i.e. when `get_node_meta(node).view_info.type == ViewType.ALLOCATED`.

    When a Node/tensor X is used as an argument to another Node/operation F,
    we say F does a reading operation on the view source under X;
    similarily, when a argument Node/tensor Y is inplace modified by F,
    we say F does a writing operation on the view source under Y.

    Along the node list as the original EASIER program runs, at different
    timings/Nodes, a view source may be modified (and maybe multiple times!).
    Those modifying timings/Nodes become barriers.
    Any operation/Node that directly reads/writes a view source, cannot be
    reordered across any barrier formed by specifically that view source.
    (i.e. crossing barrier of other view sources does not matter.)

    This Analyzer will add data dependency edges between those reader/writers
    Nodes to express such barrier-like constraints.
    """

    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        super().__init__(modules, graphs)

        # Each writer Node refreshes the status of a view source;
        # All subsequent readers, and the next writer, have
        # data dependency on this writer.
        self.src2writer: Dict[ViewSrc, Node] = {}

        self.src2readers: Dict[ViewSrc, OrderedSet[Node]] = {}

        # When a writer Node (e.g. Reducer-with-out) is never referred
        # in dataflow, it will still be added to src2writer,
        # but may not be touched anymore, e.g. no more writers/readers torch
        # the same view source of `out=x`.
        #
        # TODO currently no more dep edges will come out of that writer Node.
        # But as commented under `if_output` handlers, we may need to add
        # dep edges from such writer Nodes to the `output` Node.

    def add_data_dependency_edge(self, src: Node, dst: Node):
        if src is dst:
            # inplace ops may be both reader and writer on the same view src.
            return

        if src in dst.all_input_nodes:
            # Avoid adding a data dependency edge if a dataflow edge
            # already exists (between the two Nodes).
            return

        # Use OrderedSet to deduplicate.
        dep_inputs: OrderedSet[Node] = dst.meta.setdefault(
            KEY__DATA_DEPENDENCY_INPUTS, OrderedSet())
        dep_users: OrderedSet[Node] = src.meta.setdefault(
            KEY__DATA_DEPENDENCY_USERS, OrderedSet())

        # Since we traverse each Node in the node list once, and have avoid
        # above cases, we won't add redundant data dependency edges between
        # the same (src,dst)-pair.

        dep_inputs.add(src)
        dep_users.add(dst)

    def add_viewsrc_reader(self, view_src: ViewSrc):
        """
        Add the current node as a reader to the specified view src.
        """
        reader = self.current_node

        latest_writer = self.src2writer.get(view_src, None)
        if latest_writer is not None:
            self.add_data_dependency_edge(latest_writer, reader)

        # Other readers in the same barrier-barrier region.
        # Since the next writer will be enforced to be after these readers,
        # between these readers we do not need a order.
        readers = self.src2readers.get(view_src, None)
        if readers is None:
            readers = OrderedSet()
            self.src2readers[view_src] = readers

        readers.add(reader)

    def add_viewsrc_writer(self, view_src: ViewSrc):
        """
        Add the current node as a writer to the specified view src.
        """
        writer = self.current_node

        prev_writer: Optional[Node] = self.src2writer.get(view_src, None)
        prev_readers: Iterable[Node] = self.src2readers.get(view_src, [])

        # By concat-ing prev_write with prev_reads, we add a dedicated dep edge
        # between the two writes, even there are reads between them, e.g.
        # ```
        # n1 = setitem(a, (:), 42)
        # b  = getitem(a, (:))
        # n2 = setitem(a, (:), 43)
        # ```
        # we get 3 dep edges `n1->b, b->n2, n1->n2` even though the first 2
        # dep edges are composable.

        for prev in itertools.chain([prev_writer], prev_readers):
            if prev is None:
                continue  # may have no prev write

            # We will avoid adding a data dependency edge if a dataflow edge
            # already exists (between the two Nodes).
            # But we do not bother checking
            # if a dataflow path (i.e. many connected edges) exists.
            self.add_data_dependency_edge(prev, writer)

        # refresh the status of the view src ("add a barrier")
        self.src2writer[view_src] = writer
        self.src2readers[view_src] = OrderedSet()

    def get_view_src(self, arg: Node) -> ViewSrc:
        arg_meta = MetaV.assert_non_structured(arg)
        view_info = arg_meta.view_info

        if view_info.src is None:
            assert view_info.type == ViewType.ALLOCATED
            return arg

        else:
            assert view_info.src is not None
            return view_info.src

    def handle_any_writing_operation(self):
        """
        Handle if any writing operation into (the view src of) the argument
        pointed by `easier_is_inplace: Optional[Node]` of node current node.
        """
        inplace_arg = self.current_node.meta.get('easier_is_inplace', None)
        if inplace_arg is not None:
            assert isinstance(inplace_arg, Node)

            inplace_arg_meta = MetaV.assert_non_structured(inplace_arg)
            inplace_arg_view_info = inplace_arg_meta.view_info
            assert not (
                inplace_arg_view_info.type == ViewType.ALLOCATED
                and inplace_arg_view_info.is_multi_result_item()
            ), \
                "must have already been converted to a derived view by" \
                " operator.getitem MetaRule"

            arg_view_src = self.get_view_src(inplace_arg)

            self.add_viewsrc_writer(arg_view_src)

    def handle_reading_operation(self, arg: Node):
        """
        Handle one reading operation to (the view src of) some argument
        of the current node.

        NOTE being actually a syntactic structure, getting-item out of a
        multi-result torch op must not be regarded as a reading operation here
        and must be handled elsewhere (in `if_function_or_method`).
        """
        arg_view_src = self.get_view_src(arg)
        self.add_viewsrc_reader(arg_view_src)

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        # FX deduplicates if two esr.Module attributes refer to the same
        # esr.Tensor, so we can directly use the Node itself as ViewSrc.
        self.add_viewsrc_reader(self.current_node)

    # TODO
    # Some functions are purely manipulating the views, e.g.
    # `a2 = a[...]; a3 = a[...]`, and followed by `a[:] = 42; b = sin(a2)`.
    # As long as we ensure `a[:] and b` are kept in order,
    # it's actually safe to move `a2 a3` around.
    # We might omit adding data dependency edges on them if needed.

    def if_function_or_method(self, op_callable):
        for arg in self.current_node.all_input_nodes:
            arg_meta = get_node_meta(arg)
            if not isinstance(arg_meta, EasierTensorMeta):
                # Special case: getitem out of a multi-return op
                # the multi-return op won't be referred elsewhere.
                # However `op.getitem` also serves for tensor slicing.
                assert op_callable is operator.getitem
                continue

            self.handle_reading_operation(arg)

        # mark write after reads.
        self.handle_any_writing_operation()

    def if_call_module(self, submod: nn.Module):

        # Currently we inline both esr.Module and torch.Module, so nested
        # esr.Module calls simply become a plain one-layer graph, where
        # call_module is only about primitive Selector or Reducer.
        # However, when we switch to NOT INLINE esr.Module, we need to:
        # TODO propagate inside-esr.Module R/W dependency to outer esr.Modules
        assert type(submod) in [esr.Reducer, esr.Selector]

        for arg in self.current_node.all_input_nodes:
            MetaV.assert_non_structured(arg)
            self.handle_reading_operation(arg)

        self.handle_any_writing_operation()

    def if_output(self):
        # TODO output is strictly a syntactic element, but since FX has
        # dedicated output Node, we may add data dependency edges from output
        # to all side-effectful Nodes that do not have output Node as a
        # descendent, to ensure in subsequent graph manipulation, output will
        # not be incorrectly reordered upwards and deactivate those Nodes.
        return None


def analyze_data_dependency(modules: List[esr.Module], graphs: List[Graph]):
    """
    PyTorch operations that take Nodes/tensors as inputs and return a
    Node(itself)/tensor, are actually reading/writing the storage
    underneath those Nodes/tensors.

    The idea of "views" informs that different Nodes/tensors may refer to the
    same storage, therefore the reading/writing ops should follow the order
    of `graph.nodes: List[Node]` when we reorder the node list, even though
    thinking nodes to form a dataflow graph, in a graph-theoretical sense.

    The data dependency analysis enforces such a reading/writing order by
    adding _data dependency edges_ to the graph.

    Graph manipulation passes (like dataflow fusion) can leverage those extra
    edges (between data dependency inputs/users) as well as the original
    dataflow edges (between node inputs/users) to ensure numerical correctness
    with tensor-writing operations like `y[:]=x` or `Reducer.forward(out=x)`.
    """
    DataDependencyAnalyzer(modules, graphs).run()

    return modules, graphs
