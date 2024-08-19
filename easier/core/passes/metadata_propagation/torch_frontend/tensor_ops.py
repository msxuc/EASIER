# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from typing import Callable, Dict, Optional, Tuple, Union, Sequence, cast

import torch
from torch.fx.node import Node, Argument
from easier.core.passes.metadata_propagation import \
    EasierTensorMeta, MetadataRuleBase, metadata_rule_registry, \
    get_meta_from_ir_literal
from easier.core.passes.metadata_propagation.metadata import \
    BOOL, Role, promote_scalar_types
from easier.core.passes.metadata_propagation.utils import \
    broadcast_args, extract_shape_from_varargs, \
    Validation as V


class TensorFillRule(MetadataRuleBase):
    # Tensor.fill_ has no counterpart non-inplace op.

    # - func: fill_(Tensor input, Scalar value) -> Tensor
    # - func: fill_(Tensor input, Tensor value) -> Tensor
    def propagate(self, input, value) -> EasierTensorMeta:
        meta = V.assert_non_structured(value)
        V.equals(meta.role, Role.REPLICA)
        V.equals(meta.ndim, 0)

        self.node.meta["easier_is_inplace"] = input

        ometa = V.assert_non_structured(input)
        view_info = ometa.view_info.derive_new_view(input)
        return EasierTensorMeta(
            shape=ometa.shape, dtype=ometa.dtype, role=ometa.role,
            view_info=view_info)


metadata_rule_registry[torch.Tensor.fill_] = TensorFillRule

# Since the input is always ndim-0 replica, no extra rewriting is needed.


class RepeatRule(MetadataRuleBase):
    # NOTE `torch.Tensor.repeat` is an auto-generated Python API wrappers,
    # whose native signature is defined as:
    # `- func: repeat(Tensor self, SymInt[] repeats) -> Tensor`
    # and its Python wrapper is defined and overloaded as:
    # ```
    # def repeat(input, *_unknown_int_varargs:int)  # 1
    # def repeat(input, repeats:List[int])          # 2
    # ```
    # P.S. for native signatures with only an `int[]` arg,
    # PyTorch generates a special Python layer overload like No.1
    # for ease of invocation.
    #
    # So it can be called with arg tuple like
    # `x.r(2), x.r(2,2), x.r([2,2]), x.r(repeats=[2,2])`.
    #
    # Since we don't have FX normalization for `Tensor.repeat`,
    # we rely on `*args` to capture vararg ints,
    # and rely on kwarg `repeats` to capture the explicit list.
    def propagate(self, input,
                  repeats: Union[int, Sequence[int]], *args: int
                  ) -> EasierTensorMeta:
        repeat_nums = extract_shape_from_varargs(repeats, args)

        imeta = V.assert_non_structured(input)

        if imeta.role.is_distributed:
            V.equals(len(repeat_nums), len(imeta.shape))
            V.equals(repeat_nums[0], 1)
        else:
            # Replica-only:
            # e.g. zeros(2,3,4).repeat(5,6,7,8,9).shape
            #   == (5, 6, 7*2, 8*3, 9*4)
            V.require(len(repeat_nums) >= len(imeta.shape))

        rev_shp = []
        for d, r in itertools.zip_longest(
                reversed(imeta.shape), reversed(repeat_nums), fillvalue=1
        ):
            rev_shp.append(d * r)

        shp = tuple(reversed(rev_shp))
        return EasierTensorMeta(shape=shp, dtype=imeta.dtype, role=imeta.role)


# NOTE `repeat` has no function-variant API
# NOTE `repeat` always returns a copy
metadata_rule_registry[torch.Tensor.repeat] = RepeatRule


class ConcatRule(MetadataRuleBase):
    # - func: concat(Tensor[] tensors, int dim=0) -> Tensor
    def propagate(self, tensors, dim=0) -> EasierTensorMeta:
        V.require(len(tensors) > 0)
        imetas = [V.assert_non_structured(t) for t in tensors]
        meta0 = imetas[0]

        roles = set(m.role for m in imetas)
        V.require(len(roles) == 1)
        role, = roles

        out_ndim = meta0.ndim
        dim = range(out_ndim)[dim]
        V.require(0 <= dim < out_ndim)

        left_dimlens = set(
            (m.shape[:dim], m.shape[(dim + 1):]) for m in imetas)
        V.require(len(left_dimlens) == 1)  # also ensure ndims are the same

        if role.is_distributed:
            V.require(dim != 0)

        cat_dimlen = sum(m.shape[dim] for m in imetas)

        shp = meta0.shape[:dim] + (cat_dimlen,) + meta0.shape[(dim + 1):]
        dtype = promote_scalar_types(*(m.dtype for m in imetas))
        return EasierTensorMeta(shape=shp, dtype=dtype, role=roles.pop())


# All these are aliases and have no method-variant counterpart.
for catop in [torch.cat, torch.concat, torch.concatenate]:
    metadata_rule_registry[catop] = ConcatRule


class StackMetaRule(MetadataRuleBase):
    def propagate(self, tensors, dim=0) -> EasierTensorMeta:
        stack_n = len(tensors)
        V.require(stack_n > 0)
        imetas = [V.assert_non_structured(t) for t in tensors]
        meta0 = imetas[0]

        roles = set(m.role for m in imetas)
        V.require(len(roles) == 1)
        role, = roles

        shapes = set(m.shape for m in imetas)
        V.require(len(shapes) == 1)

        out_ndim = meta0.ndim + 1
        dim = range(out_ndim)[dim]
        V.require(0 <= dim < out_ndim)

        if role.is_distributed:
            V.require(dim != 0)

        shp = meta0.shape[:dim] + (stack_n,) + meta0.shape[dim:]
        dtype = promote_scalar_types(*(m.dtype for m in imetas))
        return EasierTensorMeta(shape=shp, dtype=dtype, role=roles.pop())


metadata_rule_registry[torch.stack] = StackMetaRule


class SqueezeMetaRule(MetadataRuleBase):
    def propagate(self, input,
                  dim: Optional[Union[int, Sequence[int]]] = None
                  ) -> EasierTensorMeta:
        imeta = V.assert_non_structured(input)
        if dim is None:
            dim = []
            for i, dimlen in enumerate(imeta.shape):
                # NOTE dimlen==0 is not affected
                if dimlen == 1:
                    dim.append(i)
        elif isinstance(dim, int):
            dim = range(imeta.ndim)[dim]
            dim = [dim]
        elif isinstance(dim, Sequence):
            raise NotImplementedError(
                "Newly added in torch 2.0, need to check the behavior")

        desc_dims = sorted(dim, reverse=True)

        if imeta.role.is_distributed:
            # seems unlikely because batchsize would hardly be 1
            V.require(not (imeta.shape[0] == 1 and desc_dims[-1] == 0))

        shp = list(imeta.shape)
        for d in desc_dims:
            shp.pop(d)

        shp = tuple(shp)

        view_info = imeta.view_info.derive_new_view(input)

        return EasierTensorMeta(shape=shp, dtype=imeta.dtype, role=imeta.role,
                                view_info=view_info)


for sqzop in [torch.squeeze, torch.Tensor.squeeze]:
    metadata_rule_registry[sqzop] = SqueezeMetaRule

    # TODO currently we don't accept torch-2.0-added `dim:Sequent[int]`


class UnsqueezeMetaRule(MetadataRuleBase):
    def propagate(self, input, dim: int) -> EasierTensorMeta:
        imeta = V.assert_non_structured(input)
        out_ndim = imeta.ndim + 1
        dim = range(out_ndim)[dim]
        V.require(0 <= dim < out_ndim)

        if imeta.role.is_distributed:
            V.require(dim != 0)

        shp = imeta.shape[:dim] + (1,) + imeta.shape[dim:]

        view_info = imeta.view_info.derive_new_view(input)

        return EasierTensorMeta(shape=shp, dtype=imeta.dtype, role=imeta.role,
                                view_info=view_info)


for unsqzop in [torch.unsqueeze, torch.Tensor.unsqueeze]:
    metadata_rule_registry[unsqzop] = UnsqueezeMetaRule


def _where_rule_core(condition, input, other) -> EasierTensorMeta:
    cond_meta = V.assert_non_structured(condition)
    V.equals(cond_meta.dtype, BOOL)
    return broadcast_args(cond_meta, input, other)


class TorchFuncWhereRule(MetadataRuleBase):
    def propagate(self, condition, input, other) -> EasierTensorMeta:
        #               ~~~~~~~~~  ~~~~~
        return _where_rule_core(condition, input, other)


metadata_rule_registry[torch.where] = TorchFuncWhereRule


class TensorMethodWhereRule(MetadataRuleBase):
    # NOTE `Tensor.where` is very special that the `self` parameter is
    # the second, so it cannot be handled by the same rule as function variant.
    def propagate(self, input, condition, other) -> EasierTensorMeta:
        #               ~~~~~  ~~~~~~~~~
        return _where_rule_core(condition, input, other)


metadata_rule_registry[torch.Tensor.where] = TensorMethodWhereRule


class TransposeMetadataRule(MetadataRuleBase):
    # - func: transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
    def propagate(self, input, dim0: int, dim1: int) -> EasierTensorMeta:
        imeta = V.assert_non_structured(input)
        dim0 = range(imeta.ndim)[dim0]
        dim1 = range(imeta.ndim)[dim1]
        if imeta.role.is_distributed:
            V.require(dim0 != 0 and dim1 != 0)

        shp = list(imeta.shape)
        tmp = shp[dim0]
        shp[dim0] = shp[dim1]
        shp[dim1] = tmp
        shp = tuple(shp)

        view_info = imeta.view_info.derive_new_view(input)

        return EasierTensorMeta(shape=shp, dtype=imeta.dtype, role=imeta.role,
                                view_info=view_info)


for tspop in [torch.transpose, torch.Tensor.transpose]:
    metadata_rule_registry[tspop] = TransposeMetadataRule


class ClampMetadataRule(MetadataRuleBase):
    def propagate(self, input, min=None, max=None) -> EasierTensorMeta:
        args = [input]
        if min is not None:
            args.append(min)
        if max is not None:
            args.append(max)

        return broadcast_args(*args)


for clampop in [torch.clamp, torch.Tensor.clamp]:
    metadata_rule_registry[clampop] = ClampMetadataRule


class DiagEmbedMetadataRule(MetadataRuleBase):
    # - func: diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1
    #                   ) -> Tensor
    def propagate(
            self, input,
            offset: int = 0, dim1: int = -2, dim2: int = -1
    ) -> EasierTensorMeta:
        imeta = V.assert_non_structured(input)
        V.require(imeta.ndim >= 1)

        # The given dim1, dim2 are specifying subspace on the resultant tensor
        out_ndim = imeta.ndim + 1
        dim1 = range(out_ndim)[dim1]
        dim2 = range(out_ndim)[dim2]
        V.require(dim1 != dim2)

        dimmin = min(dim1, dim2)
        dimmax = max(dim1, dim2)

        if imeta.role.is_distributed:
            V.require(dimmin != 0)

        # It's always the last dimension of the input tensor affected.
        out_dimlen = imeta.shape[-1] + abs(offset)
        out_shp = list(imeta.shape[:-1])
        # Insert at dimmin first then dimmax, so we don't need to adjust the
        # inserting offsets.
        out_shp.insert(dimmin, out_dimlen)
        out_shp.insert(dimmax, out_dimlen)

        return EasierTensorMeta(shape=tuple(out_shp),
                                dtype=imeta.dtype, role=imeta.role)


for diagop in [torch.diag_embed, torch.Tensor.diag_embed]:
    metadata_rule_registry[diagop] = DiagEmbedMetadataRule
