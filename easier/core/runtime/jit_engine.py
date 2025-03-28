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
    get_easier_objects, EasierInterpreter, tree_map
from easier.core.utils import EasierJitException, logger
from easier.core.runtime.dist_env import get_runtime_dist_env


class RuntimeEvaluator(EasierInterpreter):
    """
    Serves as the core to execute the fx.Graph by evaludating each Node.

    Derived 
    """
    def __init__(self, module: esr.Module, graph: Graph) -> None:
        super().__init__([module], [graph])
    
    def run(self) -> 'RuntimeEvaluator':
        # TODO we specifically limit the stackframe in the scope of `run`
        self.stackframe: dict[Node, object] = {}

        super().run()

        self.stackframe.clear()

        return self
    
    def for_each_node(self):
        val = super().for_each_node()
        self.stackframe[self.current_node] = val
    
    def _eval_args_kwargs(self):
        def _eval(x):
            if isinstance(x, Node):
                return self.stackframe[x]
            else:
                return x
        args = tuple(tree_map(v, _eval) for v in self.current_node.args)
        kwargs = {
            k: tree_map(v, _eval)
            for k, v in self.current_node.kwargs.items()
        }
        return args, kwargs


    def if_get_attr(self, submod_path: str, attr_name: str, attr_val):
        return attr_val
    
    def if_call_module(self, submod: nn.Module):
        if isinstance(submod, esr.Module):
            # no args, no return
            submod()
            return None

        elif isinstance(submod, (esr.Selector, esr.Reducer)):
            args, kwargs = self._eval_args_kwargs()
            res = submod(*args, **kwargs)
            return res
    
        else:
            assert False, 'unreachable'
    
    def if_call_function(self, function):
        args, kwargs = self._eval_args_kwargs()
        res = function(*args, **kwargs)
        return res
    
    def if_call_method(self, method_name):
        (this, *args), kwargs = self._eval_args_kwargs()

        if isinstance(this, torch.Tensor):
            # TODO are there any cases pytorch returns a non-Tensor object?
            raise EasierJitException(
                "expect a method of torch.Tensor to be called"
            )
        
        method = getattr(this, method_name)
        # `getattr` on the instance `this` already binds the method to the obj
        # so we don't pass `this` as an argument.
        res = method(*args, **kwargs)

        return res



class MetaPropEvaluator(RuntimeEvaluator):
    def for_each_node(self):
        res = super().for_each_node()

class JitEvaluator(RuntimeEvaluator):
    def if_call_module(self, submod: nn.Module):
        if isinstance(submod, GraphModule):
            # TODO check size dtype, trigger JIT
            pass
        return super().if_call_module(submod)
    
    
class EasierJitEngine:
    def __init__(self, module: esr.Module, graph: Graph) -> None:
        self.module = module
        self.graph = graph

        self.first_run = True
    
    def forward(self):
        if self.first_run:
            metaprop = MetaPropEvaluator(self.module, self.graph)
            metaprop.run()

            self.first_run = False
        
        else:
            # modules, graphs = passes.analyze_data_dependency(modules, graphs)

            # modules, graphs = passes.fuse_dataflow(modules, graphs)

            # modules, graphs = passes.generate_code(modules, backend, graphs)