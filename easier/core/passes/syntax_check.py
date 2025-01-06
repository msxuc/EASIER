# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast

import torch
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node
from easier.core.passes.utils import EasierInterpreter

from easier.core.utils import \
    EasierJitException
import easier.core.module as _EsrMod
import easier.core.runtime.modules as _EsrRuntime


class SyntaxChecker(EasierInterpreter):
    def __init__(self, modules, graphs):
        super().__init__(modules, graphs)

    def if_placeholder(self):
        raise EasierJitException(
            "easier.Module.forward() cannot have parameters"
        )

    def if_call_function(self, function):
        # Only for call_function:
        # torch.Tensor methods don't have double-underscore field "__module__".
        if function.__module__.startswith('easier.'):
            # To make Python expressions like "esr.sum(x)" FX-traceable,
            # we are facing the dilemma that all callables in `esr` Python
            # module are traced.
            # The most extreme example would be:
            # ```fxgraph
            # %compile = call_function[target=easier.core.compile](%x)
            # ```
            if function not in _EsrMod.easier_aggregators:
                raise EasierJitException(
                    f"Unexpected function call to {function}"
                )


    def if_call_module(self, submod: nn.Module):
        if isinstance(submod, _EsrMod.Module):
            # TODO will be changed in the future and allow nested calls
            raise EasierJitException(
                "Cannot have nested esr.Module calls"
            )

        elif isinstance(submod, (_EsrMod.Selector, _EsrMod.Reducer)):
            pass 

        else:
            raise EasierJitException(
                f"torch.nn module {submod} is not supported, consider using"
                " torch.nn.functional function instead"
            )


    def if_output(self):
        if self.current_node.args != (None,):
            raise EasierJitException(
                "easier.Module.forward() cannot have return value"
            )

def check_syntax(modules, graphs):
    """
    Check EASIER-specific syntax for user programs.
    """
    SyntaxChecker(modules, graphs).run()
    return modules, graphs