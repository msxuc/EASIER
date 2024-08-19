# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast

import torch
from torch.nn.modules import Module
import torch.overrides
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument
from easier.core.module import Module
from easier.core.passes.metadata_propagation.metadata import EasierTensorMeta
from easier.core.passes.utils import EasierInterpreter
from easier.core.runtime.data_loader import ATTRIBUTE_PLACEHOLDER

from easier.core.utils import \
    EasierJitException
import easier.core.module as esr

from .utils import Validation as V, get_function_variant
from .metadata import \
    EasierTensorMeta, EasierTensorMeta, Role, INT32, \
    StructuredTensorMeta, set_node_meta, \
    convert_torch_dtype_to_scalar_type, get_node_meta
from .rule_registry import MetadataRuleBase, metadata_rule_registry


def _get_rule_for(key: Union[Callable, Type[torch.nn.Module]]
                  ) -> Type[MetadataRuleBase]:
    rule_cls = metadata_rule_registry.get(key, None)
    if rule_cls is None:
        raise EasierJitException(
            f"No metadata rule registered for {key}"
        )
    return rule_cls


# A user-defined `easier.Module` cannot have output, we assign a dummy
# metadata to the `op=output` Node.
_mod_res_meta = EasierTensorMeta(shape=(), dtype=INT32, role=Role.REPLICA)


class MetadataPropagator(EasierInterpreter[StructuredTensorMeta]):
    def for_each_node(self):
        self.current_node.meta['easier_is_inplace'] = None
        meta = super().for_each_node()
        self.current_node.meta['easier_meta'] = meta

    def if_get_attr(self, submod_path, attr_name, obj) -> StructuredTensorMeta:
        shape = tuple(obj.shape)
        dtype = convert_torch_dtype_to_scalar_type(obj.dtype)

        if isinstance(obj, esr.Tensor):

            # TODO please note esr.Tensors here may be placeholders given by
            # data loaders (before dist_pass), so the properties read from
            # `obj` may be irrelevant (e.g. torch.Tensor.strides)
            #
            # if hasattr(obj, ATTRIBUTE_PLACEHOLDER):
            #     pass

            if obj.is_partition:
                meta = EasierTensorMeta(
                    shape, dtype, Role.PARTITION)
            else:
                meta = EasierTensorMeta(shape, dtype, Role.REPLICA)
        else:
            meta = EasierTensorMeta(shape, dtype, Role.REPLICA)

        return meta

    def if_function_or_method(self, callee) -> StructuredTensorMeta:
        rule_cls = _get_rule_for(callee)
        rule_inst = rule_cls(self.current_node, callee)
        return rule_inst.invoke()

    def if_call_module(self, callee: Module) -> StructuredTensorMeta:
        if isinstance(callee, esr.Module):
            # We check against the risk that during tracing-time the
            # `op=call_module` Node to `easier.Module` is used, because
            # calls to such FX "leaf modules" become `fx.Proxy`s.
            if len(self.current_node.users) > 0:
                raise EasierJitException(
                    "The result of easier.Module.forward() is always"
                    " None and cannot be used"
                )
            return _mod_res_meta

        rule_cls = _get_rule_for(type(callee))
        rule_inst = rule_cls(self.current_node, callee)
        return rule_inst.invoke()

    def if_output(self) -> StructuredTensorMeta:
        if self.current_node.args != (None,):
            raise EasierJitException(
                "easier.Module.forward() cannot have return value"
            )
        return _mod_res_meta


def propagate_metadata(modules: List[esr.Module], graphs: List[Graph]):
    MetadataPropagator(modules, graphs).run()
    return modules, graphs
