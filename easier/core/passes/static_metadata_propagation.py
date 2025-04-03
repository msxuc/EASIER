# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import enum
import math
import operator
from types import ModuleType
from typing import \
    Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast
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


KEY__STATICMETADATA_META = "easier_staticMetadata_meta"


class Role(enum.Enum):
    # Does not have batch dim
    REPLICATED = 0

    # Has batch dim, if batch size == 0 the Node is not runnable
    DISTRIBUTED = 1

    # Has batch dim, even if batch size == 0 the Node must be run
    HALO_EXCHANGER = 2

@dataclasses.dataclass(frozen=True, eq=True)
class StaticTensorMeta:
    role: Role
    
    # For Role.REPLICATED, batch_size is always 0
    batch_size: int

    def __post_init__(self):
        if self.role == Role.REPLICATED:
            assert self.batch_size == 0
        elif self.role == Role.DISTRIBUTED or self.role == Role.HALO_EXCHANGER:
            assert self.batch_size >= 0
        else:
            assert f'bad value {self}'

_replica = StaticTensorMeta(Role.REPLICATED, 0)
_halo_exchanger = StaticTensorMeta(Role.HALO_EXCHANGER, 0)

class StaticMetadataPropagator(EasierInterpreter[StaticTensorMeta]):
    """
    This class propagates Roles, in a relatively simpler way than the
    tensor_grouping pass.
    And RolePropagator will handle EASIER-inserted Nodes like
    the reordering Selectors for Reducers or HaloExchangers, which do not
    have EasierTensorGroup objects related.

    This class should only be used on well-validated and rewritten Graphs,
    so we don't need to validate with details again.
    """
    def __init__(self, modules, graphs) -> None:
        super().__init__(modules, graphs)

    def for_each_node(self) -> StaticTensorMeta:
        meta = super().for_each_node()
        self.current_node.meta[KEY__STATICMETADATA_META] = meta
        return meta
    
    def if_get_attr(
        self, submod_path: str, attr_name: str, attr_val
    ) -> StaticTensorMeta:
        if isinstance(attr_val, _EsrMod.Tensor) and attr_val.is_partition:
            return StaticTensorMeta(Role.DISTRIBUTED, attr_val.shape[0])
        else:
            return _replica

    def if_function_or_method(
        self, function_or_method_name
    ) -> StaticTensorMeta:
        # TODO currently it's the _EsrMod.reduce (before
        # easier.runtime.all_gather_into_tensor) to be the conversion point
        # frrom DIST to REPLICA.
        # However, when it comes to future refinement e.g. using all_reduce,
        # the conversion point becomes that all_reduce Node.
        # And perhaps dataflow_dist pass is more suitable to figure out the
        # Role, and it can store the Role e.g. using a special wrapper function
        # easier.all_reduce and the information remains in the dump.
        in_metas = set(
            get_static_metadata(n) for n in self.current_node.all_input_nodes
        )
        dist_in_metas = in_metas - set([_replica])

        if function_or_method_name in _EsrMod.easier_aggregators:
            assert len(dist_in_metas) == 1, \
                "EASIER aggregator input should be distributed"
            return _replica
        
        assert len(dist_in_metas) <= 1
        if len(dist_in_metas) == 1:
            meta = dist_in_metas.pop()
            assert meta.role is not Role.HALO_EXCHANGER
            return meta
        else:
            return _replica
    
    
    def if_call_module(self, submod: nn.Module) -> StaticTensorMeta:
        in_metas = set(
            get_static_metadata(n) for n in self.current_node.all_input_nodes
        )

        if isinstance(submod, _EsrMod.Module):
            assert len(in_metas) == 0
            return _replica
        
        if isinstance(submod, _EsrMod.Selector):
            return StaticTensorMeta(Role.DISTRIBUTED, submod.idx.shape[0])

        if isinstance(submod, _EsrMod.Reducer):
            return StaticTensorMeta(Role.DISTRIBUTED, submod.n)

        if isinstance(submod, HaloExchanger):
            return StaticTensorMeta(
                Role.HALO_EXCHANGER, 0  #submod.output_batch_size
            )

        assert False, f'unreachable {submod}'
    
    def if_output(self) -> StaticTensorMeta:
        return _replica


def propagate_static_metadata(modules: List[_EsrMod.Module], graphs: List[Graph]):
    logger.info(str(graphs[0]))
    propagator = StaticMetadataPropagator(modules, graphs)
    propagator.run()

    return modules, graphs


def get_static_metadata(node: Node) -> StaticTensorMeta:
    """
    After role propagation, every Node should have a Role assigned.
    """
    role = node.meta[KEY__STATICMETADATA_META]
    return role
