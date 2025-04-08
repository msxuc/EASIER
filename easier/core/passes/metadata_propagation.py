# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import enum
import math
import operator
from types import ModuleType
from typing import \
    Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeAlias, Union, cast
from typing_extensions import Literal
import more_itertools
import os

import torch
from torch import nn
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.proxy import Proxy
from torch.fx.graph import Graph

from easier.core import module as _EsrMod
from easier.core.passes.utils import \
    FX, get_easier_objects, EasierInterpreter, tree_map
from easier.core.utils import EasierJitException, logger
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger


KEY__METADATA_STATIC = "easier_metadata_staticNodeMeta"
KEY__METADATA_RUNTIME = 'easier_metadata_runtimeTensorMeta'


class Role(enum.Enum):
    # Does not have batch dim
    REPLICATED = 0

    # Has batch dim, if batch size == 0 the Node is not runnable
    # (but if it's HaloExchanger, even if batch size == 0 it must be run)
    DISTRIBUTED = 1


@dataclasses.dataclass(frozen=True, eq=True)
class StaticNodeMeta:
    """
    Static metadata is associated with the Node,
    because during ahead-of-time compilation (or before the first run
    in JitEngine) we don't know if a Node returns a nested structure
    (e.g. `maxval, maxpos = torch.max(dim=1)`).

    EASIER will assume, in the runtime, all nested tensors will have
    the same role/batch_size info.
    """
    role: Role

    # For Role.REPLICATED, batch_size is always 0
    batch_size: int

    def __post_init__(self):
        if self.role == Role.REPLICATED:
            assert self.batch_size == 0
        elif self.role == Role.DISTRIBUTED:
            assert self.batch_size >= 0
        else:
            assert f'bad value {self}'


@dataclasses.dataclass(frozen=True, eq=True)
class RuntimeTensorMeta:
    """
    Runtime metadata is associated with a Tensor instance.

    In contrast to StaticNodeMeta, at runtime we may encounter with cases
    that a single PyTorch operator returns a tuple/list of Tensors
    (e.g. `maxval, maxpos = torch.max(dim=1)`).

    Therefore we need to be aware of the potential nested structure of
    runtime metadata.
    """
    role: Role
    shape: Tuple[int, ...]
    dtype: torch.dtype


StructuredTensorMeta: TypeAlias = Union[
    RuntimeTensorMeta,
    Sequence['StructuredTensorMeta']
]

_replica = StaticNodeMeta(Role.REPLICATED, 0)


class StaticMetadataPropagator(EasierInterpreter[StaticNodeMeta]):
    """
    This class propagates StaticNodeMeta, in a relatively simpler way than the
    tensor_grouping pass.
    And RolePropagator will handle EASIER-inserted Nodes like
    the reordering Selectors for Reducers or HaloExchangers, which are not
    aware or accepted by tensor_grouping.

    This class should only be used on well-validated and rewritten Graphs,
    so we don't need to validate with details again.
    """

    def __init__(self, modules, graphs) -> None:
        super().__init__(modules, graphs)

    def for_each_node(self) -> StaticNodeMeta:
        meta = super().for_each_node()
        self.current_node.meta[KEY__METADATA_STATIC] = meta
        return meta

    def if_get_attr(
        self, submod_path: str, attr_name: str, attr_val
    ) -> StaticNodeMeta:
        if isinstance(attr_val, _EsrMod.Tensor) and attr_val.is_partition:
            return StaticNodeMeta(Role.DISTRIBUTED, attr_val.shape[0])
        else:
            return _replica

    def if_function_or_method(
        self, function_or_method_name
    ) -> StaticNodeMeta:
        # TODO currently it's the _EsrMod.reduce (before
        # easier.runtime.all_gather_into_tensor) to be the conversion point
        # frrom DIST to REPLICA.
        # However, when it comes to future refinement e.g. using all_reduce,
        # the conversion point becomes that all_reduce Node.
        # And perhaps dataflow_dist pass is more suitable to figure out the
        # Role, and it can store the Role e.g. using a special wrapper function
        # easier.all_reduce and the information remains in the dump.
        in_metas = set(
            get_static_node_metadata(n) for n in self.current_node.all_input_nodes
        )
        dist_in_metas = in_metas - set([_replica])

        if function_or_method_name in _EsrMod.easier_aggregators:
            assert len(dist_in_metas) == 1, \
                "EASIER aggregator input should be distributed"
            return _replica

        assert len(dist_in_metas) <= 1
        if len(dist_in_metas) == 1:
            meta = dist_in_metas.pop()
            return meta
        else:
            return _replica

    def if_call_module(self, submod: nn.Module) -> StaticNodeMeta:
        if isinstance(submod, _EsrMod.Module):
            return _replica

        if isinstance(submod, _EsrMod.Selector):
            return StaticNodeMeta(Role.DISTRIBUTED, submod.idx.shape[0])

        if isinstance(submod, _EsrMod.Reducer):
            # NOTE if we need to inspect the input StaticNodeMetas,
            # we must be aware of the case `Reducer.foward(halos_concat, out)`
            # that the two input metas are not the same, therefore we'll have
            # `len(in_metas) == 2`
            return StaticNodeMeta(Role.DISTRIBUTED, submod.n)

        if isinstance(submod, HaloExchanger):
            return StaticNodeMeta(
                Role.DISTRIBUTED, submod.output_batch_size
            )

        assert False, 'unreachable'

    def if_output(self) -> StaticNodeMeta:
        return _replica


def propagate_static_node_metadata(
    modules: List[_EsrMod.Module], graphs: List[Graph]
):
    propagator = StaticMetadataPropagator(modules, graphs)
    propagator.run()

    return modules, graphs


def get_static_node_metadata(node: Node) -> StaticNodeMeta:
    """
    After static metadata propagation, every Node should be assigned.
    """
    role = node.meta[KEY__METADATA_STATIC]
    return role


def set_runtime_tensor_metadata(node: Node, tensor_meta: StructuredTensorMeta):
    node.meta[KEY__METADATA_RUNTIME] = tensor_meta


def get_runtime_tensor_metadata(node: Node) -> StructuredTensorMeta:
    return node.meta[KEY__METADATA_RUNTIME]
