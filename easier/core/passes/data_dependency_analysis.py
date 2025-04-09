# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
import itertools
import operator
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast, FrozenSet, TYPE_CHECKING
from typing_extensions import TypeAlias

import torch
import torch.overrides
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument, map_arg
from easier.core.passes.metadata_propagation import \
    StaticNodeMeta

from easier.core.utils import logger, EasierJitException
import easier.core.module as esr

from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, tree_map, get_easier_tensors

if TYPE_CHECKING:
    from easier.core.runtime.jit_engine import RuntimeValue

KEY__DATA_DEPENDENCY_INPUTS = 'easier_dataDependency_inputs'
KEY__DATA_DEPENDENCY_USERS = 'easier_dataDependency_users'


def get_data_dependency_inputs(node: Node) -> List[Node]:
    # Return a clone to avoid manipulating the node meta dict.
    return list(node.meta.get(KEY__DATA_DEPENDENCY_INPUTS, []))


def get_data_dependency_users(node: Node) -> List[Node]:
    # Return a clone to avoid manipulating the node meta dict.
    return list(node.meta.get(KEY__DATA_DEPENDENCY_USERS, []))


def _collect_addrs(
    runtime_vals: Union['RuntimeValue', list]
) -> OrderedSet[int]:
    addrs = OrderedSet()

    def _insert(x):
        if isinstance(x, torch.Tensor):
            # ignoring strides/offsets
            addr: int = x.storage().data_ptr()
            addrs.add(addr)
        
    _ = tree_map(runtime_vals, _insert)
    return addrs


class DataDependencyAnalyzer(EasierInterpreter):
    """
    When a Node/tensor X is used as an argument to another Node/operation F,
    we say F does a reading operation on the memory under X;
    similarily, when a argument Node/tensor Y is inplace modified by F,
    or the operation F allocates the memory of Y,
    we say F does a writing operation on the memory under Y.

    Along the node list as the original EASIER program runs, at different
    timings/Nodes, a memory may be modified (and maybe multiple times!).
    Those modifying timings/Nodes become barriers.
    Any operation/Node that directly reads/writes a memory, cannot be
    reordered across any barrier formed by specifically that memory.
    (i.e. crossing barrier of other memory does not matter.)

    This Analyzer will add data dependency edges between those reader/writers
    Nodes to express such barrier-like constraints.

    We present a memory by its memory address, ignoring strides/offsets
    introduced by the PyTorch view operations e.g. X[:, :, 2]
    Also, the addresses are in the same memory space i.e. CPU or CUDA memory.
    """
    def __init__(
        self,
        modules: Sequence[esr.Module],
        graphs: Sequence[Graph],
        stackframe: Dict[Node, 'RuntimeValue']
    ):
        """
        Args:
        -   stackframe:
                the snapshot of the stackframe from the runtime,
                no memory space should be reused during the execution
                (i.e. not used tensors should not be released)
        """
        assert len(modules) == len(graphs) == 1, \
            "One module/graph at a time, avoid adding dep edges cross graphs"
            # otherwise, 1) provide multiple stackframes,
            # 2) clear add2readers/write when switching current_graph.

        super().__init__(modules, graphs)

        self.stackframe = stackframe
        
        # TODO this requires, in the runtime, intermediate results are
        # not released even if it's no longer used.
        # Otherwise, there will be conflicts for simple `int` addresses.
        self.addr2srcnode: Dict[int, Node] = {}

        for node, val in stackframe.items():
            for addr in _collect_addrs(val):
                if addr not in self.addr2srcnode:
                    self.addr2srcnode[addr] = node


        # Each writer Node refreshes the status of a view source;
        # All subsequent readers, and the next writer, have
        # data dependency on this writer.
        #
        # The source Node itself is NOT??? included.
        #
        # When a writer Node (e.g. Reducer-with-out) is never referred
        # in dataflow, it will still be added to src2writer,
        # but may not be touched anymore, e.g. no more writers/readers torch
        # the same view source of `out=x`.
        self.addr2writer: Dict[int, Node] = {}

        self.addr2readers: Dict[int, OrderedSet[Node]] = {}


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
            KEY__DATA_DEPENDENCY_INPUTS, OrderedSet()
        )
        dep_users: OrderedSet[Node] = src.meta.setdefault(
            KEY__DATA_DEPENDENCY_USERS, OrderedSet()
        )

        # Since we traverse each Node in the node list once, and have avoid
        # above cases, we won't add redundant data dependency edges between
        # the same (src,dst)-pair.

        dep_inputs.add(src)
        dep_users.add(dst)

    def add_reader_dependency(self, arg_addr: int):
        """
        Add the current node as a reader to the memory at `arg_addr`.
        """
        reader = self.current_node

        latest_writer = self.addr2writer[arg_addr]
        self.add_data_dependency_edge(latest_writer, reader)

        # Other readers in the same barrier-barrier region.
        # Since the next writer will be enforced to be after these readers,
        # between these readers we do not need a order.
        readers = self.addr2readers.setdefault(arg_addr, OrderedSet())
        readers.add(reader)

    def add_writer_dependency(self, res_addr: int):
        """
        Add the current node as a writer to the memory at `res_addr`.
        Including the operation that allocates the memory.
        """
        writer = self.current_node

        # If current_node is the operation that allocates, we get no prev
        # writers.
        prev_writer: Optional[Node] = self.addr2writer.get(res_addr, None)
        prev_readers: Iterable[Node] = self.addr2readers.get(res_addr, [])

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
        self.addr2writer[res_addr] = writer
        self.addr2readers[res_addr] = OrderedSet()

    def for_each_node(self):
        # Some ops may take multiple input writable tensors, and the result
        # will be a tuple of those tensors themselves.
        #
        # JitEngine also executes and records `operator.setitem` in this way.
        #
        # TODO any FX-traceable torch ops violate this convention? Or any way
        # to detect if the violation happens?
        #
        res_addrs: OrderedSet[int] = _collect_addrs(
            self.stackframe[self.current_node]
        )
        arg_addrs: OrderedSet[int] = _collect_addrs([
            self.stackframe[arg_node]
            for arg_node in self.current_node.all_input_nodes
        ])

        for arg_addr in arg_addrs:
            self.add_reader_dependency(arg_addr)
        for res_addr in res_addrs:
            self.add_writer_dependency(res_addr)

        # Handle more specific scenarios like nested esr.Module calls
        return super().for_each_node()

    # TODO
    # Some functions are purely manipulating the views, e.g.
    # `a2 = a[...]; a3 = a[...]`, and followed by `a[:] = 42; b = sin(a2)`.
    # As long as we ensure `a[:] and b` are kept in order,
    # it's actually safe to move `a2 a3` around.
    # We might omit adding data dependency edges on them if needed.

    def if_call_module(self, submod: nn.Module):
        if isinstance(submod, esr.Module):
            # A nested esr.Module call has no input/output, data dependency
            # may occur through esr.Tensor instances the inner Module
            # shares/writes.
            # Simply set read/write dependencies on ALL esr.Tensors in module.
            #
            # TODO we may identify the concrete RECURSIVE intersection set
            # of esr.Tensors
            tensors: Dict[esr.Tensor, list] = get_easier_tensors(
                [self.current_module]
            )
            param_addrs = _collect_addrs(list(tensors))

            for param_addr in param_addrs:
                self.add_reader_dependency(param_addr)
                self.add_writer_dependency(param_addr)

    def if_output(self):
        # TODO output is strictly a syntactic element, but since FX has
        # dedicated output Node, we may add data dependency edges from output
        # to all side-effectful Nodes that do not have output Node as a
        # descendent, to ensure in subsequent graph manipulation, output will
        # not be incorrectly reordered upwards and deactivate those Nodes.
        return


def analyze_data_dependency(
    modules: List[esr.Module],
    graphs: List[Graph],
    stackframe: Dict[Node, 'RuntimeValue']
):
    """
    PyTorch operations that take Nodes/tensors as inputs and return a
    Node(itself)/tensor, are actually reading/writing the storage/memory
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
    DataDependencyAnalyzer(modules, graphs, stackframe).run()

    return modules, graphs
