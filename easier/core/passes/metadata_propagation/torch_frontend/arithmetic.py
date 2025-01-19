# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Callable, Dict, Tuple, Union

import torch
from torch.fx.node import Node, Argument

from easier.core.passes.metadata_propagation.rule_registry import \
    MetadataRuleBase, metadata_rule_registry, inplace_version
from easier.core.passes.metadata_propagation.metadata import \
    FLOAT32, EasierTensorMeta, Role
from easier.core.passes.metadata_propagation.utils import \
    get_method_variant, broadcast_args, Validation as V


#
# Unary operators that keep input dtype
#


class UnaryArithRule(MetadataRuleBase):
    def propagate(self, input) -> EasierTensorMeta:
        return V.assert_non_structured(input)


for uniop in [
    operator.neg
]:
    metadata_rule_registry[uniop] = UnaryArithRule


for uniop in [
    torch.neg, torch.negative,
    torch.abs,
    torch.sign,
]:
    metadata_rule_registry[uniop] = UnaryArithRule
    metadata_rule_registry[get_method_variant(uniop)] = UnaryArithRule


for inplace_uniop, uniop in [
    (torch.Tensor.neg_, torch.neg),
    (torch.Tensor.negative_, torch.negative),
    (torch.Tensor.abs_, torch.abs),
    (torch.Tensor.sign_, torch.sign),
]:
    metadata_rule_registry[inplace_uniop] = inplace_version(uniop)


#
# Unary operators that convert integers to float32
#


class UnaryFloatArithRule(MetadataRuleBase):
    def propagate(self, input) -> EasierTensorMeta:
        dummy_float = EasierTensorMeta(
            shape=(), dtype=FLOAT32, role=Role.REPLICA)
        return broadcast_args(input, dummy_float)


for unifop in [
    torch.sqrt, torch.square, torch.exp,
]:
    metadata_rule_registry[unifop] = UnaryFloatArithRule
    metadata_rule_registry[get_method_variant(unifop)] = UnaryFloatArithRule


for inplace_unifop, unifop in [
    (torch.Tensor.sqrt_, torch.sqrt),
    (torch.Tensor.square_, torch.square),
    (torch.Tensor.exp_, torch.exp),
]:
    metadata_rule_registry[inplace_unifop] = inplace_version(unifop)


#
# Binary operators that keep input dtype
#


class BinaryArithRule(MetadataRuleBase):
    def propagate(self, input, other, alpha=None) -> EasierTensorMeta:
        if alpha is not None:
            V.require(self.callee in [torch.add, torch.sub])
            alpha_meta = V.assert_non_structured(alpha)
            V.equals(alpha_meta.shape, ())

        return broadcast_args(input, other)


for binop in [
    operator.add, operator.sub, operator.mul,
    operator.floordiv,
]:
    metadata_rule_registry[binop] = BinaryArithRule


for binop in [
    torch.add, torch.sub, torch.mul,
    torch.floor_divide,
]:
    metadata_rule_registry[binop] = BinaryArithRule
    metadata_rule_registry[get_method_variant(binop)] = BinaryArithRule


for inplace_binop, binop in [
    (torch.Tensor.add_, torch.add),
    (torch.Tensor.sub_, torch.sub),
    (torch.Tensor.mul_, torch.mul),
    (torch.Tensor.floor_divide_, torch.floor_divide),
]:
    metadata_rule_registry[inplace_binop] = inplace_version(binop)


#
# Binary operators that convert integers to float32
#


class BinaryFloatArithRule(MetadataRuleBase):
    def propagate(self, input, other) -> EasierTensorMeta:
        dummy_float = EasierTensorMeta(
            shape=(), dtype=FLOAT32, role=Role.REPLICA)
        return broadcast_args(input, other, dummy_float)


for binfop in [
    operator.truediv,
]:
    metadata_rule_registry[binfop] = BinaryFloatArithRule


for binfop in [
    torch.true_divide, torch.div, torch.divide,
]:
    metadata_rule_registry[binfop] = BinaryFloatArithRule
    metadata_rule_registry[get_method_variant(binfop)] = BinaryFloatArithRule


for inplace_binop, binop in [
    (torch.Tensor.add_, torch.add),
    (torch.Tensor.sub_, torch.sub),
    (torch.Tensor.mul_, torch.mul),
    (torch.Tensor.floor_divide_, torch.floor_divide),
    (torch.Tensor.true_divide_, torch.true_divide),
    (torch.Tensor.div_, torch.div),
    (torch.Tensor.divide_, torch.divide),
]:
    metadata_rule_registry[inplace_binop] = inplace_version(binop)


# `pow()` doesn't enforce the output to be float, using the default
# dtype promotion policy.
class PowArithRule(MetadataRuleBase):
    def propagate(self, input, exponent) -> EasierTensorMeta:
        # NOTE `torch.pow` specifies the parameter name `exponent`.
        return broadcast_args(input, exponent)


for powop in [operator.pow, torch.pow, torch.Tensor.pow]:
    metadata_rule_registry[powop] = PowArithRule
