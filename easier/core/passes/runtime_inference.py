# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
    get_easier_objects, EasierInterpreter, tree_map
from easier.core.utils import EasierJitException, logger
from easier.core.runtime.dist_env import get_runtime_dist_env


KEY__RUNTIMEINFERENCE_ROLE = "easier_runtimeInference_role"

class Role(enum.Enum):
    REPLICATED = 0
    DISTRIBUTED = 1

class RolePropagator(EasierInterpreter[Role]):
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

        self.node2role: Dict[Node, Role] = {}
    
    def for_each_node(self) -> Role:
        role = super().for_each_node()
        self.current_node.meta[KEY__RUNTIMEINFERENCE_ROLE] = role
        return role
    
    def if_get_attr(self, submod_path: str, attr_name: str, attr_val) -> Role:
        if isinstance(attr_val, _EsrMod.Tensor) and attr_val.is_partition:
            return Role.DISTRIBUTED
        else:
            return Role.REPLICATED

    def if_function_or_method(self, function_or_method_name) -> Role:
        # TODO currently it's the esr.reduce (before
        # easier.runtime.all_gather_into_tensor) to be the conversion point
        # frrom DIST to REPLICA.
        # However, when it comes to future refinement e.g. using all_reduce,
        # the conversion point becomes that all_reduce Node.
        # And perhaps dataflow_dist pass is more suitable to figure out the
        # Role, and it can store the Role e.g. using a special wrapper function
        # easier.all_reduce and the information remains in the dump.
        in_roles = set(
            self.node2role[n] for n in self.current_node.all_input_nodes
        )

        if function_or_method_name in _EsrMod.easier_aggregators:
            assert in_roles == set([Role.DISTRIBUTED]), \
                "EASIER aggregator input should be distributed"
            return Role.REPLICATED
        
        if Role.DISTRIBUTED in in_roles:
            return Role.DISTRIBUTED
        else:
            return Role.REPLICATED
    
    
    def if_call_module(self, submod: nn.Module) -> Role:
        in_roles = set(
            self.node2role[n] for n in self.current_node.all_input_nodes
        )

        if isinstance(submod, _EsrMod.Module):
            assert len(in_roles) == 0
            return Role.REPLICATED

        # Selector Reducer HaloExchanger
        assert in_roles == set([Role.DISTRIBUTED]), \
            "call_module Node has all inputs distributed"
        return Role.DISTRIBUTED
    
    def if_output(self) -> Role:
        return Role.REPLICATED


def propagate_roles(modules: List[_EsrMod.Module], graphs: List[Graph]):
    propagator = RolePropagator(modules, graphs)
    propagator.run()

    return modules, graphs


def get_node_role(node: Node) -> Role:
    """
    After role propagation, every Node should have a Role assigned.
    """
    role = node.meta[KEY__RUNTIMEINFERENCE_ROLE]
    return role