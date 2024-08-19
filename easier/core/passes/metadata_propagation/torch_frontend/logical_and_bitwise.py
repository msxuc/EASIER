# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Callable, Dict, Tuple, Union

import torch
from torch.fx.node import Argument

from easier.core.passes.metadata_propagation import \
    EasierTensorMeta, MetadataRuleBase, metadata_rule_registry, \
    get_meta_from_ir_literal
from easier.core.passes.metadata_propagation.metadata import BOOL
from easier.core.passes.metadata_propagation.utils import \
    get_method_variant, broadcast_args, Validation as V


#
# Logical operators, always return bool tensors
#


class LogicalResultUnaryArithRule(MetadataRuleBase):
    def propagate(self, input) -> EasierTensorMeta:
        imeta = V.assert_non_structured(input)
        return EasierTensorMeta(shape=imeta.shape,
                                dtype=BOOL,
                                role=imeta.role)

# NOTE operator.not_ (i.e. Python keyword `not`) cannot be used, because
# it converts a singleton bool Tensor into a Python bool.


for logical_uniop in [
    torch.logical_not
]:
    metadata_rule_registry[logical_uniop] = LogicalResultUnaryArithRule
    metadata_rule_registry[get_method_variant(logical_uniop)] = \
        LogicalResultUnaryArithRule


class LogicalResultBinaryArithRule(MetadataRuleBase):
    def propagate(self, input, other) -> EasierTensorMeta:
        return broadcast_args(input, other, dtype=BOOL)


for logical_binop in [
    operator.le, operator.lt, operator.ge, operator.gt,
    operator.eq, operator.ne
]:
    metadata_rule_registry[logical_binop] = LogicalResultBinaryArithRule


for logical_binop in [
    torch.le, torch.lt, torch.ge, torch.gt, torch.eq, torch.ne,
    torch.logical_and, torch.logical_or, torch.logical_xor,
    # NOTE `operator.and_` etc. are not logical operators,
    # but bitwise operators that keep input dtypes.
]:
    metadata_rule_registry[logical_binop] = LogicalResultBinaryArithRule
    metadata_rule_registry[get_method_variant(logical_binop)] = \
        LogicalResultBinaryArithRule


#
# Bitwise operators, always take integer tensors
#


class BitwiseUnaryRule(MetadataRuleBase):
    def propagate(self, input) -> EasierTensorMeta:
        imeta = V.assert_non_structured(input)
        V.require(imeta.dtype.is_integer)
        return imeta


for bitwise_uniop in [
    operator.inv, operator.invert,
    torch.bitwise_not, torch.Tensor.bitwise_not
]:
    metadata_rule_registry[bitwise_uniop] = BitwiseUnaryRule


class BitwiseBinaryRule(MetadataRuleBase):
    def propagate(self, input, other) -> EasierTensorMeta:
        imeta1 = V.assert_non_structured(input)
        V.require(imeta1.dtype.is_integer)
        imeta2 = V.assert_non_structured(other)
        V.require(imeta2.dtype.is_integer)
        return broadcast_args(imeta1, imeta2)


for bitwise_binop in [
    operator.and_, operator.or_, operator.xor
]:
    metadata_rule_registry[bitwise_binop] = BitwiseBinaryRule


for bitwise_binop in [
    torch.bitwise_and, torch.bitwise_or, torch.bitwise_xor
]:
    metadata_rule_registry[bitwise_binop] = BitwiseBinaryRule
    metadata_rule_registry[get_method_variant(bitwise_binop)] = \
        BitwiseBinaryRule
