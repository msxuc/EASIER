# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast
from typing_extensions import TypeAlias

import torch
import torch.overrides
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument, map_arg
from easier.core.passes.metadata_propagation.metadata import \
    EasierTensorMeta, Role, get_node_meta

from easier.core.utils import \
    logger, EasierJitException
import easier.core.module as esr

from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, \
    normalize_reducer_call_into_args, normalize_selector_call_into_args, \
    FX, DisjointSet, tree_map


KEY__TENSORGROUPING_GROUP = "easier_tensorGrouping_group"


# The objects that define the (layout of) tensors, and we are constructing
# equivalency between them:
# - The esr.Tensors are `.dist=='partition'` only
# - esr.Reducer/Selector instances are representing the groups of their results
EasierTensorDef: TypeAlias = Union[esr.Tensor, esr.Reducer, esr.Selector]


@dataclass
class EasierTensorGroup:
    tensor_defs: OrderedSet[EasierTensorDef]
    n: int

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)


def _get_tensordef_batch_size(tensordef: EasierTensorDef):
    if isinstance(tensordef, esr.Tensor):
        n = tensordef.shape[0]
    elif isinstance(tensordef, esr.Selector):
        n = tensordef.idx.shape[0]
    elif isinstance(tensordef, esr.Reducer):
        n = tensordef.n
    else:
        assert False, "unreachable"

    return n


class TensorGrouper(EasierInterpreter[Optional[EasierTensorDef]]):
    """
    Each Node is interpreted to at most one single TensorDef.

    That is, even if the Node is a multiple-output operation,
    all its output distributed tensors belong to the same TensorGroup.
    """

    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        super().__init__(modules, graphs)

        # Inputs to Selector/Reducer are equivalent based on the
        # instances of Selector/Reducer modules. Even those inputs
        # never meet each other in batched operations.
        # And we only store one such Node as the representative.
        self.selector_input_equiv: Dict[torch.nn.Module, EasierTensorDef] = {}
        self.reducer_input_equiv: Dict[torch.nn.Module, EasierTensorDef] = {}

        # Some Nodes, like those for replicas, do not have a _TensorDef
        self.node2def: Dict[Node, Optional[EasierTensorDef]] = {}

        # The equivalency explicitly excludes None (i.e. replica Nodes)
        self.group_dset: DisjointSet[EasierTensorDef] = \
            DisjointSet(equal=lambda x, y: x is y)

    def set_equivalent(self, defs: Sequence[Optional[EasierTensorDef]]
                       ) -> Optional[EasierTensorDef]:
        """
        Set equivalency over TensorGroups (i.e. the non-Nones).

        Return an arbitrary TensorGroup within the equivalent set, or None if
        no TensorGroups are input.
        """
        dset: List[EasierTensorDef] = list(
            filter(lambda d: d is not None, defs)
        )  # type: ignore

        self.group_dset.union(*dset)
        if len(dset) > 0:
            return dset.pop()
        else:
            return None

    def for_each_node(self):
        tensordef: Optional[EasierTensorDef] = super().for_each_node()
        self.node2def[self.current_node] = tensordef

    def if_get_attr(self, submod_path, attr_name, attr_val
                    ) -> Optional[EasierTensorDef]:
        if isinstance(attr_val, esr.Tensor) and attr_val.is_partition:
            return self.set_equivalent([attr_val])
        else:
            return None

    def if_function_or_method(self, op_callable):
        # Only if the resultant metadata (even part of the multi-output)
        # indicates it's distributed do we
        # assume the output TensorGroup equivalency.
        output_roles: Set[Role] = set()

        def _collect_output_roles(x):
            assert isinstance(x, EasierTensorMeta)
            output_roles.add(x.role)
            return None
        node_meta = get_node_meta(self.current_node)
        _ = tree_map(node_meta, _collect_output_roles)

        if Role.PARTITION in output_roles:
            # We must explicitly exclude e.g. replica Nodes otherwise we are
            # building wrong equivalency.
            input_defs = [self.node2def[arg]
                          for arg in self.current_node.all_input_nodes]
            # And it's ok to pass Nones to `set_equivalent`.
            return self.set_equivalent(input_defs)
        else:
            return None

    def if_call_module(self, module: torch.nn.Module):
        args = self.current_node.args
        kwargs = self.current_node.kwargs

        inplace_out_def: Optional[EasierTensorDef] = None
        if isinstance(module, esr.Selector):
            input_node = \
                normalize_selector_call_into_args(*args, **kwargs)
            in_equivs = self.selector_input_equiv

        elif isinstance(module, esr.Reducer):
            input_node, opt_inplace_out_node = \
                normalize_reducer_call_into_args(*args, **kwargs)

            if isinstance(opt_inplace_out_node, Node):
                inplace_out_def = self.node2def[opt_inplace_out_node]

            in_equivs = self.reducer_input_equiv

        else:
            raise EasierJitException(
                f'{type(module)} is not supported to appear in'
                ' tensor grouping'
            )

        assert isinstance(input_node, Node)
        input_def = self.node2def[input_node]
        assert input_def is not None

        if module in in_equivs:
            prev_input_tensordef = in_equivs[module]
            self.set_equivalent([prev_input_tensordef, input_def])

            # Check input batch shape consistency
            # (mainly for Selector)
            imeta: EasierTensorMeta = get_node_meta(input_node)  # type: ignore
            n = _get_tensordef_batch_size(prev_input_tensordef)
            if n != imeta.shape[0]:
                raise EasierJitException(
                    f"A tensor with batch size {imeta.shape[0]} is passed to"
                    f" {module.__class__.__name__}"
                    f" at {self.current_node.target} which was previously"
                    f" taking tensors with batch size {n} as input")

        else:
            in_equivs[module] = input_def

        # module instance itself is always the TensorDef for call_module Nodes
        self.set_equivalent([module, inplace_out_def])

        return module


def group_tensors(modules: List[esr.Module], graphs: List[Graph]):
    """
    Group tensors by:
    -   if they are operands to the same batched operation.
    -   if they are inputs/ouputs of the same Selector/Reducer instance.

    NOTE this pass must be run after metadata propagation.
    """
    tensor_grouper = TensorGrouper(modules, graphs)
    tensor_grouper.run()

    # NOTE only after we have seen all IRs can we get the final equivalency

    def2grp: Dict[EasierTensorDef, EasierTensorGroup] = {}
    for defset in tensor_grouper.group_dset.get_ordered_sets():
        tdef0, *_ = defset
        grp = EasierTensorGroup(
            tensor_defs=defset,
            n=_get_tensordef_batch_size(tdef0))
        for tdef in defset:
            def2grp[tdef] = grp

            # Store the TensorGroup into TensorDef instance
            # Replica tensors will have this field being its default value None
            tdef.easier_tensor_group = grp

    # Store the TensorGroup into Node.meta
    class NodeMetaAssigner(EasierInterpreter):
        def for_each_node(self):
            node = self.current_node
            optdef: Optional[EasierTensorDef] = tensor_grouper.node2def[node]
            if optdef is not None:
                optgrp = def2grp[optdef]
            else:
                optgrp = None
            node.meta[KEY__TENSORGROUPING_GROUP] = optgrp
    NodeMetaAssigner(modules, graphs).run()

    return modules, graphs


def get_node_tensor_group(node: Node) -> Optional[EasierTensorGroup]:
    """
    If an TensorDef is given for a Node, it's applied to all output distributed
    tensors of this Node.
    If None is given, it means this Node does not have a distributed output.
    """
    grp = node.meta[KEY__TENSORGROUPING_GROUP]
    return grp
