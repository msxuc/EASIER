# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Sequence

import torch.fx
from torch.nn.modules import Module

from easier.core.jit import EasierTracer
import easier.core.module as esr
from easier.core.passes.utils import EasierInterpreter, get_easier_objects
from easier.core.utils import EasierJitException


class Jvp(esr.Module):
    """
    """
    def __init__(self):
        super().__init__()

        inputs: Sequence[esr.Tensor]
        outputs: Sequence[esr.Tensor]

        vector: Sequence[esr.Tensor]
        product: Sequence[esr.Tensor]
        # TODO how about tangents_in tangents_out?
    
    def forward(self):
        raise EasierJitException()


"""
TODO
-   Calculate connectivity between each op to a specified input/ouput,
    ops that aren't connected are not transformed to forward AD.
    -   all "sources/sinks" are in the middle of the graph.
    -   Nodes to carry dual numbers are connected to at least one input and at least one output.
"""

class JvpTransformer(EasierInterpreter):
    """
    Recursive transformation from a pair of Module/Graph for primal calculation
    to a Module/Graph for jvp.

    The recursion happens on JvpTransformer rather than `easier.jvp()`,
    the result of this is that there won't be explicit easier.Tensors to
    store tangents at the boundary of nested easier.Modules.
    """
    def __init__(self, module: esr.Module):
        graph = EasierTracer().trace(module)

        super().__init__([module], [graph])


    def if_call_module(self, submod: Module):
        if isinstance(submod, esr.Module):
            # Nested easier.Module, must be JVP-ed.
            JvpTransformer(submod).run()


        return super().if_call_module(submod)

def jvp(
    module: esr.Module,
    inputs: Sequence[esr.Tensor],
    outputs: Sequence[esr.Tensor]
) -> Jvp:

    # TODO the resultant Jvp module should be disconnected from `module`
    # in a way that get_easier_object(jvp) does not include moudle.
    # TODO assign meaningful names to:
    # - Jvp.inputs, like `x` if input is InputModule.x
    # - Jvp.vector, like `tan_x`


    jvp_transformer = JvpTransformer(module).run()

    return Jvp()