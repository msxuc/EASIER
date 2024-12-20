# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast

import torch
import torch.overrides
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument

from easier.core.passes.utils import \
    FX, fx_normalize_function_variant_into_kwargs, tree_map, EasierJitException
import easier.core.module as esr

from .utils import Validation as V, get_function_variant
from .metadata import \
    EasierTensorMeta, EasierTensorMeta, Role, INT32, \
    StructuredTensorMeta, get_node_meta


class MetadataRuleBase:

    normalize_to_kwargs_only: bool = False

    def propagate(self, *args, **kwargs) -> StructuredTensorMeta:
        raise NotImplementedError("Derived class should implement this")

    def ensure_batch_size_consistency(
        self, input_dist_metas: List[EasierTensorMeta]
    ) -> None:
        """
        override-able.
        """
        if len(input_dist_metas) > 0:
            batchsizes = set(vm.shape[0] for vm in input_dist_metas)
            V.require(len(batchsizes) == 1)

    def __init__(self, node, callee):
        self.node: Node = node

        # - callable `torch.f` for op='call_function'
        # - callable `torch.Tensor.f` for op='call_method'
        # - instance of subtype of `torch.nn.Module` for op='call_module'
        self.callee = callee

    def invoke(self) -> StructuredTensorMeta:
        #
        # Pre-propagation checks:
        #

        # 1. Ensure no mix of vertex tensors and edge tensors.
        inmetas: List[StructuredTensorMeta] = \
            list(map(get_node_meta, self.node.all_input_nodes))
        input_tensor_metas: List[EasierTensorMeta] = []

        # Input node may have structured metadata
        # e.g. some input is a tensor list.
        def _collect_input_tensor_meta(x):
            assert isinstance(x, EasierTensorMeta)
            input_tensor_metas.append(x)
            return None
        _ = tree_map(inmetas, _collect_input_tensor_meta)

        input_dist_metas = list(filter(
            lambda m: m.role.is_distributed, input_tensor_metas))

        # 2. Ensure consistency of n-ary batched operations
        self.ensure_batch_size_consistency(input_dist_metas)

        #
        # Invoke operator-specific propagation
        #
        def _naive_propagate():
            return self.propagate(*self.node.args, **self.node.kwargs)

        if self.node.op == FX.CALL_FUNCTION:

            if self.normalize_to_kwargs_only:
                kwargs: Dict[str, Argument] \
                    = fx_normalize_function_variant_into_kwargs(
                        self.node.target, self.node.args, self.node.kwargs
                )
                prop_res = self.propagate(**kwargs)

            else:
                prop_res = _naive_propagate()

        elif self.node.op == FX.CALL_METHOD:

            if self.normalize_to_kwargs_only:
                # Get the corresponding function variant for a method variant
                # operator, e.g. `torch.Tensor.neg` -> `torch.neg`.
                # For those like `torch.Tensor.repeat` which don't have
                # function variants, this will throw.
                func_variant = get_function_variant(self.callee)

                kwargs: Dict[str, Argument] \
                    = fx_normalize_function_variant_into_kwargs(
                        func_variant, self.node.args, self.node.kwargs
                )

                prop_res = self.propagate(**kwargs)

            else:
                prop_res = _naive_propagate()

        else:
            prop_res = _naive_propagate()

        #
        # Post-propagation:
        #
        return prop_res


metadata_rule_registry: Dict[Union[Callable, Type[torch.nn.Module]],
                             Type[MetadataRuleBase]] = {}


def inplace_version(origin_func) -> Type[MetadataRuleBase]:

    class InplaceRule(MetadataRuleBase):
        def propagate(self, input, *args, **kwargs) -> EasierTensorMeta:
            rule = metadata_rule_registry[origin_func](self.node, self.callee)
            broadcasted_meta = rule.invoke()
            broadcasted_meta = V.assert_non_structured(broadcasted_meta)

            # We are firstly using non-inplace-version rule to infer, then
            # we need to ensure that inferred result is write-able
            # (broadcastable) into the
            # container specified by `input`. And because `input` is also
            # involved in the previous inference, the resultant metadata
            # should be equal to the original metadata of `input`.
            # NOTE unlike `setitem`, inplace ops like `sub_` generally do not
            # allow operands to have prefix "length-1 dims".
            container_meta = V.assert_non_structured(input)
            V.equals(broadcasted_meta.shape, container_meta.shape)
            V.equals(broadcasted_meta.dtype, container_meta.dtype)
            V.equals(broadcasted_meta.role, container_meta.role)
            # Don't count the view info. Only validate shape/dtype/role as
            # the operands are broadcasted into the container.

            self.node.meta['easier_is_inplace'] = self.node.args[0]

            # Typically programmers do not use inplace ops return values,
            # the fact is the result of an inplace op is always a view
            # on the real storage of the written-into tensor (which is probably
            # a view, too).
            view_info = container_meta.view_info.derive_new_view(input)
            ret_meta = EasierTensorMeta(
                shape=container_meta.shape, dtype=container_meta.dtype,
                role=container_meta.role,
                view_info=view_info
            )

            return ret_meta

    return InplaceRule
