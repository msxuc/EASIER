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


def create_zero_arg_val(
    raw_node_arg: Union[Node, Sequence[Node]]
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    """
    `raw_node_arg` is an argument of raw Graph Node, it may be a nested
    structure,
    e.g. torch.cat Node may have `args[0] == [x1, x2, x3]`.
    e.g. getitem may have `args == (input, (slice(), 3, slice()))`
    
    The result may also be a nested structure of many zero tensors.

    Always on CPU.
    """
    def _make(x):
        if isinstance(x, Node):
            meta = get_node_meta(x)
            assert isinstance(meta, RuntimeTensorMeta), \
                "Value of ARG Node cannot be nested structure"

            return torch.zeros(meta.shape, dtype=meta.dtype)
        else:
            return x

    return tree_map(raw_node_arg, _make)  # type: ignore


def wrap_operator_specific_kwargs_normalizer(
    normalizer: Callable[[Callable, tuple, dict], Dict[str, FxArg]]
) -> Callable[[Callable, tuple, dict], Dict[str, FxArg]]:
    # Python syntactic operator like 1+a may have operands being ints, which
    # is unsupported by standard FX normalizer.
    def _wrap(function_variant, args: tuple, kwargs: dict):
        assert len(args) <= 2
        assert len(kwargs) == 0
        mapping = {}
        f_args = []
        for arg in args:
            # getitem is not allowed here, so only possibilities are int/float
            if isinstance(arg, (int, float)):
                f_arg = torch.tensor([1.0])
                mapping[f_arg] = arg
                arg = f_arg
            f_args.append(arg)

        d = normalizer(function_variant, tuple(f_args), kwargs)

        # Revert f_arg above back to raw arg
        d_raw = {}
        for k, v in d.items():
            if v in mapping:
                v = mapping[v]
            d_raw[k] = v
        
        return d_raw

    return _wrap


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
            diff_rule_registry, differentiabilities, getitem_aux_kw

        if function in diff_rule_registry:
            rule_cls = diff_rule_registry[function]
            rule = rule_cls(
                self.current_node, function, self._fake_eval_meta_ctor
            )

            out_meta = rule.invoke_output_meta()
            set_node_meta(self.current_node, out_meta)
        
        else:
            wrap_normalizer = lambda f: f

            if getattr(operator, function.__name__, None) is function:
                if function is operator.getitem:
                    # Special path, see differentiabilities[operator.getitem]
                    function = getitem_aux_kw
                elif function is operator.truediv:
                    function = torch.div  # torch does not have truediv
                    wrap_normalizer = wrap_operator_specific_kwargs_normalizer
                else:
                    function = getattr(torch, function.__name__)
                    wrap_normalizer = wrap_operator_specific_kwargs_normalizer

            dfbs = differentiabilities[function]
            for dfb in dfbs:
                # TODO this is actually shared by all Differentiability
                # overloadings.
                raw_node_normalized_kwargs: Dict[str, FxArg] = \
                    (
                        wrap_normalizer(dfb.kwargs_normalizer)
                    )(
                        function,
                        self.current_node.args,
                        self.current_node.kwargs
                    )  # type: ignore
                if dfb.all_param_names() == set(
                    raw_node_normalized_kwargs.keys()
                ):
                    break
            else:
                assert False, \
                    "Failed to resolve overloading:" \
                    f" with Differentiabilities {dfbs}," \
                    f" got {raw_node_normalized_kwargs}"
            
            kwvals = {
                k: tree_map(
                    v, create_zero_arg_val
                ) if not isinstance(v, FxConst.__args__) else v
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

