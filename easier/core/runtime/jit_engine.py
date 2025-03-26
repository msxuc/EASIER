# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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

import easier as esr
from easier.core import passes
from easier.core import module as _EsrMod
from easier.core.passes.utils import \
    get_easier_objects, EasierInterpreter
from easier.core.utils import EasierJitException, logger
from easier.core.runtime.dist_env import get_runtime_dist_env


class RuntimeInterpreter(EasierInterpreter):
    """
    Serves as the core to execute the fx.Graph by evaludating each Node.

    Derived 
    """
    def __init__(self, modules, graphs) -> None:
        super().__init__(modules, graphs)
    
    def for_each_node(self):
        val = super().for_each_node()

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        return attr_val
    
    def if_call_module(self, submod: nn.Module):
        return super().if_call_module(submod)
    
    def if_call_function(self, function):
        return super().if_call_function(function)
    
    def if_call_method(self, method_name):
        return super().if_call_method(method)
    
    
class EasierJitEngine(EasierInterpreter):
    def __init__(self, module: esr.Module, graph: Graph) -> None:
        super().__init__([module], [graph])
    
    def forward(self):
        pass