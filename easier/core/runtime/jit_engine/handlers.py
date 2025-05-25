# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import enum
from typing import Dict, List, Union, cast, Callable

import torch
from torch.fx.node import Node
from torch.fx.graph import Graph

import easier as esr
from easier.core.passes.utils import FX
from easier.core.utils import EasierJitException
from easier.core.runtime.jit_engine.values import RuntimeValue


class PreprocessDecision(enum.Enum):
    # The whole preprocessing pipeline should go on
    CONTINUE = enum.auto()

    # Break the preprocessing pipeline,
    # skip successive Handlers and Node evaluation,
    # jit_skipped indicator object will be used as the runtime value.
    SKIP_EVAL = enum.auto()

    # Break the preprocessing pipeline, skip successive Handlers,
    # immediately evaluate the Node using current args/kwargs.
    GOTO_EVAL = enum.auto()

    # P.S. Generally speaking, derived Handlers should be designed in a way
    # that can gently continue, even previous Handlers does not explicitly
    # request GOTO_EVAL.
    # TODO this may be a hint that we may need a new enum ACTIVE to
    # differentiate from CONTINUE for Handlers that really modify the args
    # but do not break the pipeline.


class NodeHandlerBase:
    def __init__(self):
        #
        # Context variables, the lifetime is the scope of `Module.forward()`
        #
        # Because JIT time passes may change the module and the graph, we don't
        # use Handler.__init__() to bind them. JitEngine will bind them.
        # =================
        self.current_module: esr.Module
        self.current_graph: Graph
        self.stackframe: Dict[Node, RuntimeValue]

        #
        # Context variables, the lifetime is the execution period of a Node.
        # =================
        self.current_node: Node

        # When postprocess() gets run, the implementation could access the
        # resultant decision of its preprocess() subprocedure.
        self.preprocess_decision: PreprocessDecision

    def enter_forward(self) -> None:
        """
        Graph-level states of derived Handlers should be initialized here.
        """
        pass

    def exit_forward(self) -> None:
        """
        Graph-level states of derived Handlers should be cleaned up here.
        """
        pass

    def preprocess(
        self, args: List[RuntimeValue], kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        """
        A registered derived Handler can inspect and modify `args/kwargs`
        in-place.
        The final `args/kwargs` will be used to evaluate the Node.

        All Handlers' preprocess() will be run in sequence.

        Node-level states of derived Handlers should be initialized
        at the beginning of overridden preprocess().

        Args:
        -   args, kwargs:
            The mutable argument collection to inspect and modify.
        """
        root = self.current_module
        node = self.current_node

        if node.op == FX.GET_ATTR:
            path = cast(str, node.target)
            submod_path, _sep, attr_name = path.rpartition(".")
            submod = root.get_submodule(submod_path)
            obj = getattr(submod, attr_name)

            if not isinstance(obj, torch.Tensor):
                raise EasierJitException(
                    "Currently we can only reference"
                    " torch.Tensor and subtypes"
                )

            assert len(args) == len(kwargs) == 0
            decision = self.preprocess_get_attr(submod_path, attr_name, obj)

        elif node.op == FX.CALL_FUNCTION:
            assert callable(node.target)
            decision = self.preprocess_call_function(node.target, args, kwargs)

        elif node.op == FX.CALL_METHOD:
            assert isinstance(node.target, str)
            decision = self.preprocess_call_method(node.target, args, kwargs)

        elif node.op == FX.CALL_MODULE:
            submod_path = cast(str, node.target)
            callee = root.get_submodule(submod_path)
            decision = self.preprocess_call_module(callee, args, kwargs)

        elif node.op == FX.OUTPUT:
            decision = self.preprocess_output()

        else:
            assert False, f"Unexpected FX Node op {node.op}"

        return decision

    def preprocess_get_attr(
        self, submod_path: str, attr_name: str, attr_val
    ) -> PreprocessDecision:
        return PreprocessDecision.CONTINUE

    def preprocess_call_function(
        self,
        function: Callable,
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        return self.preprocess_function_or_method(function, args, kwargs)

    def preprocess_call_method(
        self,
        method_name: str,
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        return self.preprocess_function_or_method(method_name, args, kwargs)

    def preprocess_function_or_method(
        self,
        function_or_method_name: Union[Callable, str],
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        return PreprocessDecision.CONTINUE

    def preprocess_call_module(
        self,
        submod: torch.nn.Module,
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        return PreprocessDecision.CONTINUE

    def preprocess_output(self) -> PreprocessDecision:
        return PreprocessDecision.CONTINUE

    def postprocess(
        self,
        res: RuntimeValue,
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> RuntimeValue:
        """
        A registered derived Handler can inspect the Node evaluation result
        `res`,
        together with the read-only, final `args/kwargs` that are used to
        evaluate the Node.

        All Handlers' postprocess() will be run in the reversed order.

        Node-level states of derived Handlers should be cleaned up
        at the end of overridden postprocess().

        Args:
        -   res:
            Postprocessed result value from the previous postprocessor

        -   args, kwargs:
            The read-only, final arguments resulted from the whole
            preprocessing pipeline.

            NOTE if the derived Handler want to keep the arguments it sees
            on its own preprocessing step, it needs to copy the list/dict
            in the preprocess().

        Returns:
        -   the modified result for Node evaluation.
        """
        return res
