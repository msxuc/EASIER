# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import operator
from types import EllipsisType
from typing import Callable, Dict, Sequence, TypeAlias, Union, cast

import torch
from torch.fx import Graph, Node, GraphModule
from torch.fx.node import BaseArgumentTypes as _FxConstBase
from torch.fx.node import Argument as FxArg
from torch.nn.modules import Module

import easier.core.module as esr
from easier.core.passes.tensor_grouping import get_node_tensor_group
from easier.core.passes.utils import FX, EasierInterpreter, tree_map, \
    fx_normalize_function_variant_into_kwargs, \
    normalize_reducer_call_into_args, normalize_selector_call_into_args
from easier.core.runtime.metadata import Role, RuntimeTensorMeta, \
    get_node_meta, set_node_meta


# e.g. int, float, dtype, device, slice, range, etc.
FxConst: TypeAlias = Union[_FxConstBase, slice, range, EllipsisType, None]


class PrimalMetaPropagator(EasierInterpreter):
    def _fake_eval_meta_ctor(self, shape, dtype):
        # Special function needed by get_value_runtime_info, to provide
        # Role in TensorMeta.
        ng = get_node_tensor_group(self.current_node)
        if ng is None:
            role = Role.REPLICATED

        else:
            role = Role.DISTRIBUTED
            shape = (1000,) + shape[1:]

        return RuntimeTensorMeta(role, shape, dtype)

    def _create_zero_val(
        self, raw_node_arg: Union[Node, Sequence[Node]]
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        `raw_node_arg` is an argument of raw Graph Node, it may be a nested
        structure,
        e.g. torch.cat Node may have `args[0] == [x1, x2, x3]`.
        
        The result may also be a nested structure of many zero tensors.

        Always on CPU.
        """
        def _make(x):
            assert isinstance(x, Node), \
                "In a list, Node and scalar are not expected to be mixed"

            meta = get_node_meta(x)
            assert isinstance(meta, RuntimeTensorMeta), \
                "Value of arg Node cannot be nested structure"

            return torch.zeros(meta.shape, dtype=meta.dtype)

        return tree_map(raw_node_arg, _make)  # type: ignore
    
    def if_get_attr(self, submod_path: str, attr_name: str, attr_val) -> None:
        # Avoid circle import
        from easier.core.runtime.jit_engine.jit_engine import \
            get_value_runtime_info
        
        runtime_meta = get_value_runtime_info(
            self.current_node, attr_val, self._fake_eval_meta_ctor
        )
        set_node_meta(self.current_node, runtime_meta)
    
    def if_call_function(self, function: Callable) -> None:
        if function is operator.getitem:
            container, item_i = self.current_node.args
            assert isinstance(container, Node)
            imeta = get_node_meta(container)

            if isinstance(imeta, Sequence):
                assert isinstance(item_i, int)
                item_meta = imeta[item_i]
                set_node_meta(self.current_node, item_meta)
                return
        
        self._handle_operation(function)
        
    def _handle_operation(self, function: Callable):
        from easier.core.autodiff.autodiff_rule import \
    Differentiability, RequiredParam, \
    diff_rule_registry, differentiabilities

        if function in diff_rule_registry:
            rule_cls = diff_rule_registry[function]
            rule = rule_cls(
                self.current_node, function, self._fake_eval_meta_ctor
            )

            out_meta = rule.invoke_output_meta()
            set_node_meta(self.current_node, out_meta)
        
        else:
            if getattr(operator, function.__name__, None) is function:
                if function is operator.truediv:
                    function = torch.div  # torch does not have truediv
                else:
                    function = getattr(torch, function.__name__)

            raw_node_normalized_kwargs: Dict[str, FxArg] = \
                fx_normalize_function_variant_into_kwargs(
                    function, self.current_node.args, self.current_node.kwargs
                )  # type: ignore
            dfbs = differentiabilities[function]
            for dfb in dfbs:
                if dfb.all_param_names() == set(
                    raw_node_normalized_kwargs.keys()
                ):
                    break
            else:
                assert False, \
                    "Failed to resolve overloading:" \
                    f" with Differentiabilities {dfbs}," \
                    f" got {raw_node_normalized_kwargs}"
            
            raw_node_diff_args: Dict[
                str, Union[FxConst, Node, Sequence[Node]]
            ] = {
                p: raw_node_normalized_kwargs[p] for p in dfb.diffable_params
            }  # type: ignore

            kwvals = {
                k: tree_map(
                    v, self._create_zero_val
                ) if isinstance(v, Node) else v
                for k, v in raw_node_normalized_kwargs.items()
            }
            fake_res = function(**kwvals)
            
            from easier.core.runtime.jit_engine.jit_engine import \
                get_value_runtime_info
            out_meta = get_value_runtime_info(
                self.current_node, fake_res, self._fake_eval_meta_ctor
            )
            set_node_meta(self.current_node, out_meta)
        
    def if_call_method(self, method_name: str):
        function = getattr(torch.ops.aten, method_name)
        self._handle_operation(function)


    def if_call_module(self, submod: Module):
        if isinstance(submod, esr.Module):
            raise NotImplementedError()
            # Nested easier.Module, must be JVP-ed.
            # sub_jvp_transformer = JvpTransformer(self.root_jvp, submod).run()

        # TODO make Selector/Reducer rules.

        elif isinstance(submod, esr.Selector):
            input = normalize_selector_call_into_args(
                *self.current_node.args, **self.current_node.kwargs
            )
            assert isinstance(input, Node)
            
            # NOTE the fake batch size is not affected by S.idx.shape[0]
            set_node_meta(self.current_node, get_node_meta(input))

        elif isinstance(submod, esr.Reducer):
            if submod.reduce != 'sum':
                raise NotImplementedError()

            input, out = normalize_reducer_call_into_args(
                *self.current_node.args, **self.current_node.kwargs
            )
            assert isinstance(input, Node)
            assert out is None or isinstance(out, Node)

            # NOTE the fake batch size is not affected by S.idx.shape[0]
            set_node_meta(self.current_node, get_node_meta(input))

        else:
            assert False, 'unreachable'


class _TorchFuncGraphSimplifer(EasierInterpreter):
    def if_call_method(self, method_name: str):
        assert False, "torch.func.jvp() should decomposite all Tensor methods"
    
    def if_call_module(self, submod: Module):
        assert False, "EASIER modules shouldn't be handled by torch.func.jvp()"
    
    def if_call_function(self, function):
        if function in [
            torch.ops.aten.alias.default
        ]:
            x = cast(Node, self.current_node.args[0])
            self.current_node.replace_all_uses_with(x)

            self.current_graph.erase_node(self.current_node)
        
        if function in [
            torch.ops.aten.is_same_size.default,
            torch.ops.aten._has_same_storage_numel.default,
        ]:
            assert len(self.current_node.users) == 0

            self.current_graph.erase_node(self.current_node)


def simplify_torchfunc_fx_graph(jvp_gm: GraphModule) -> Graph:
    """
    The Graph (both FX or TorchScript) from torch.func.jvp() will have many
    internal, annotation-only Nodes from torch.func, for EASIER AD usage
    these Nodes are not needed.


    The raw Graph are (likely) generated by tracing with forward-AD layer
    AutogradMeta::set_fw_grad in
    $PYTORCH_BASE/torch/csrc/autograd/autograd_meta.cpp

    The removed functions like is_same_size etc. above does not do any checks
    so it's safe to remove -- they are simply traced because they are
    calls on Tensors.


    To check how the raw Graph looks like we can run this codesnippet:
    ```
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    x = torch.rand(5, 5, 5)
    p = torch.zeros(5, 5, 5)
    i = torch.zeros(5, dtype=torch.int64)

    def fun(x: torch.Tensor, p: torch.Tensor):
        return x.index_add(0, i, p)

    def myjvp(x, p, tx, tp):
        res, jvp = torch.func.jvp(fun, (x, p), (tx, tp))
        return res, jvp

    gm = make_fx(myjvp)(x, p, x.clone(), p.clone())
    print(gm.graph)
    ```
    """
    simp_g = Graph()
    out_v = simp_g.graph_copy(jvp_gm.graph, {})
    simp_g.output(out_v)

    _TorchFuncGraphSimplifer([jvp_gm], [simp_g]).run()  # type: ignore

    return simp_g

