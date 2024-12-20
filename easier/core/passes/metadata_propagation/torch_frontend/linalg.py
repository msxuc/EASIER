# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Callable, Dict, List, Set, Tuple, Union

import torch
from torch.fx.node import Node, Argument

from easier.core.passes.metadata_propagation import \
    Role, EasierTensorMeta, MetadataRuleBase, metadata_rule_registry
from easier.core.passes.metadata_propagation.metadata import \
    ScalarType, View, ViewType, get_node_meta, promote_scalar_types
from easier.core.passes.metadata_propagation.utils import \
    Validation as V, \
    broadcast_and_validate_shapes, promote_and_validate_roles

from easier.core.passes.utils import FX
from easier.core.utils import logger


# - func: linalg_svd(Tensor A, bool full_matrices=True, *, str? driver=None)
#         -> (Tensor U, Tensor S, Tensor Vh)
class SvdRule(MetadataRuleBase):
    def propagate(
            self, A, full_matrices: bool, **_unused
    ) -> Tuple[EasierTensorMeta, EasierTensorMeta, EasierTensorMeta]:
        imeta = V.assert_non_structured(A)
        V.equals(imeta.role, Role.REPLICA)

        m, n = imeta.shape
        k = min(m, n)

        if full_matrices:
            U_shp = (m, m)
            V_shp = (n, n)
        else:
            U_shp = (m, k)
            V_shp = (k, n)

        return (
            EasierTensorMeta(
                shape=U_shp, dtype=imeta.dtype, role=Role.REPLICA,
                view_info=View(ViewType.ALLOCATED, (self.node, 0))
            ),
            EasierTensorMeta(
                shape=(k,), dtype=imeta.dtype, role=Role.REPLICA,
                view_info=View(ViewType.ALLOCATED, (self.node, 1))
            ),
            EasierTensorMeta(
                shape=V_shp, dtype=imeta.dtype, role=Role.REPLICA,
                view_info=View(ViewType.ALLOCATED, (self.node, 2))
            ),
        )


# NOTE There is also a `torch.svd` but deprecated in favor of linalg.svd
metadata_rule_registry[torch.linalg.svd] = SvdRule


def _check_or_promote_scalar_types(
        check_same_dtype: bool,
        x: Union[EasierTensorMeta, ScalarType],
        y: Union[EasierTensorMeta, ScalarType],
) -> ScalarType:
    if isinstance(x, EasierTensorMeta):
        x = x.dtype
    if isinstance(y, EasierTensorMeta):
        y = y.dtype

    if check_same_dtype:
        # torch.dot, torch.matmul requires this too.
        dtype = V.equals(x, y)
    else:
        dtype = promote_scalar_types(x, y)

    return dtype


def _dot_rule_core(
        meta1: EasierTensorMeta, meta2: EasierTensorMeta,
        check_same_dtype: bool = True
) -> EasierTensorMeta:
    V.require(meta1.ndim == 1 and meta2.ndim == 1)
    V.equals(meta1.shape[0], meta2.shape[0])

    role = V.equals(meta1.role, meta2.role)
    V.equals(role, Role.REPLICA)

    dtype = _check_or_promote_scalar_types(check_same_dtype, meta1, meta2)
    return EasierTensorMeta(shape=(), dtype=dtype, role=Role.REPLICA)


def _mm_rule_core(
        meta1: EasierTensorMeta, meta2: EasierTensorMeta,
        check_same_dtype: bool = True
) -> EasierTensorMeta:
    V.require(meta1.ndim == 2 and meta2.ndim == 2)

    V.equals(meta2.role, Role.REPLICA)
    out_role = meta1.role

    m, n1 = meta1.shape
    n2, p = meta2.shape
    V.equals(n1, n2)

    dtype = _check_or_promote_scalar_types(check_same_dtype, meta1, meta2)
    return EasierTensorMeta(shape=(m, p), dtype=dtype, role=out_role)


def _mv_rule_core(
        meta1: EasierTensorMeta, meta2: EasierTensorMeta,
        check_same_dtype: bool = True
) -> EasierTensorMeta:
    # (m, n) @ (n,) = (m,)
    V.require(meta1.ndim == 2 and meta2.ndim == 1)

    V.equals(meta2.role, Role.REPLICA)
    out_role = meta1.role

    m, n1 = meta1.shape
    n2, = meta2.shape
    V.equals(n1, n2)

    dtype = _check_or_promote_scalar_types(check_same_dtype, meta1, meta2)
    return EasierTensorMeta(shape=(m,), dtype=dtype, role=out_role)


class MatmulMetadataRule(MetadataRuleBase):
    def propagate(self, input, other) -> EasierTensorMeta:
        R"""
        This table illustrates which ndim pair is handled by which branch:
        ```
             \  ndim2     
        ndim1 \ 0 1 2 3 ...
              0 x x x x x
              1 x a c e
              2 x d b e
              3 x e e e
            ... x       e
        ```
        """
        meta1 = V.assert_non_structured(input)
        meta2 = V.assert_non_structured(other)
        ndim1 = meta1.ndim
        ndim2 = meta2.ndim

        V.require(ndim1 >= 1 and ndim2 >= 1)

        # We don't try to merge those conditions even their underlying
        # behaviors can be merged,
        # to match the description of the PyTorch doc.

        if ndim1 == 1 and ndim2 == 1:
            # dot product, returns a scalar
            return _dot_rule_core(meta1, meta2, check_same_dtype=True)

        elif ndim1 == 2 and ndim2 == 2:
            # matrix product
            return _mm_rule_core(meta1, meta2, check_same_dtype=True)

        elif ndim1 == 1 and ndim2 == 2:
            # (m,) @ (m,n) = (n,)
            # `m` can be a distributed batch
            m1, = meta1.shape
            m2, n = meta2.shape
            m = V.equals(m1, m2)

            input_role = V.equals(meta1.role, meta2.role)
            V.equals(input_role, Role.REPLICA)

            dtype = V.equals(meta1.dtype, meta2.dtype)
            return EasierTensorMeta(shape=(n,), role=Role.REPLICA, dtype=dtype)

        elif ndim1 == 2 and ndim2 == 1:
            # (m,n) @ (n,) = (m,)
            return _mv_rule_core(meta1, meta2, check_same_dtype=True)

        # Note the remaining possibilities like `ndim1 in [1,2]`.
        else:
            # - ndim1 in [1,2] and ndim2 >= 3
            # - ndim1 >= 3 and ndim2 in [1,2]
            # - ndim1 >= 3 and ndim2 >= 3
            #
            # (..., n, m) @ (..., m, p) = (..., n, p) and
            # - two `...` parts may be different but broadcastable
            # - when one operand ndim==1 i.e.
            #   missing `n` (dim-1 of the left operand)
            #        or `p` (dim-2 of the right operand),
            #   length-1 dim is padded on those places, and removed after.
            return self._on_nd_bmm_with_dim_padding(meta1, meta2)

    def _on_nd_bmm_with_dim_padding(
            self,
            meta1: EasierTensorMeta, meta2: EasierTensorMeta
    ) -> EasierTensorMeta:
        ndim1 = meta1.ndim
        ndim2 = meta2.ndim
        shp1 = meta1.shape
        shp2 = meta2.shape

        max_ndim = max(ndim1, ndim2)
        popdim = None

        # At least one operand has ndim >= 3, and the heading dims are
        # for broadcast.
        for meta in [meta1, meta2]:
            if meta.role.is_distributed:
                V.equals(meta.ndim, max_ndim)
                # Batch sizes are checked in RuleBase.

        if ndim1 == 1:
            # (/*1,*/ m,) @ (..., k, m, n) = (..., k, /*1,*/ n)
            shp1 = (1,) + shp1
            popdim = -2

        elif ndim2 == 1:
            # (..., k, m, n) @ (n, /*1,*/) = (..., k, m /*,1*/)
            shp2 = shp2 + (1,)
            popdim = -1

        padshp1 = (1,) * (max_ndim - len(shp1)) + shp1
        padshp2 = (1,) * (max_ndim - len(shp2)) + shp2
        m, n1 = padshp1[-2:]
        n2, p = padshp2[-2:]
        V.equals(n1, n2)
        batchdims = broadcast_and_validate_shapes(padshp1[:-2], padshp2[:-2])
        shp = list(batchdims + (m, p))

        if popdim is not None:
            # Remove the padded dim, if any.
            shp.pop(popdim)
        shp = tuple(shp)

        out_role = promote_and_validate_roles(meta1.role, meta2.role)
        dtype = V.equals(meta1.dtype, meta2.dtype)
        return EasierTensorMeta(shape=shp, dtype=dtype, role=out_role)


def _rewrite_mm_permute_core(node: Node, input, other):
    # arg2 must be a replica

    # original computation: (nv, m) @ (m, n) = (nv, n)
    #                                 ~~~~~~
    # permuted computation: (m, nv) @ (m, n) = (n, nv)
    #                                 ~~~~~~
    # rewriting: transpose the RHS and swap operands:
    #                       (n, m) @ (m, nv) = (n, nv)
    with node.graph.inserting_before(other.next):
        other_T = node.graph.call_function(
            torch.permute,
            (other, (1, 0)), {}
        )

    # Node.op (may be 'call_method') and .target can remain the same.

    node.args = (other_T, input)
    node.kwargs = {}


def _rewrite_mv_permute_core(node: Node, input, other):
    # arg2 must be a replica

    # original computation: (nv, m) @ (m,) = (nv,)
    #                                 ~~~~
    # permuted computation: (m, nv) @ (m,) = (nv,)
    #                                 ~~~~
    # rewriting: swap operands:
    #                       (m,) @ (m, nv) = (nv,)

    # The rewritten result is no longer a valid `torch.mv`
    node.op = FX.CALL_FUNCTION
    node.target = torch.matmul

    node.args = (other, input)
    node.kwargs = {}


for matmulop in [operator.matmul, torch.matmul, torch.linalg.matmul]:
    metadata_rule_registry[matmulop] = MatmulMetadataRule


def _get_einsum_spec(
        equation: str, imetas: List[EasierTensorMeta]
) -> Tuple[
        Tuple[List[str], str],  # ([in_str, ...], out_str)
        Dict[str, int]
]:
    equation = equation.replace(' ', '')  # remove possible whitespaces
    # For e.g. 'a,b->' out_str is an empty str.
    _left, out_str = equation.split('->')
    V.require(',' not in out_str)
    in_strs = _left.split(',')
    V.equals(len(in_strs), len(imetas))

    d: Dict[str, int] = {}
    for imeta, in_str in zip(imetas, in_strs):
        V.equals(len(in_str), imeta.ndim)
        for dimlen, in_char in zip(imeta.shape, in_str):
            if in_char in d:
                V.equals(dimlen, d[in_char])
            else:
                d[in_char] = dimlen

    return (in_strs, out_str), d


class EinsumMetadataRule(MetadataRuleBase):
    # NOTE `torch.einsum` is exposed to PyTorch Python frontend as
    # `def einsum(*args) -> torch.Tensor: ...`
    # so arguments can be passed positionally only.
    def propagate(self, equation: str, *tensors) -> EasierTensorMeta:
        imetas = [V.assert_non_structured(t) for t in tensors]

        (in_strs, out_str), d = _get_einsum_spec(equation, imetas)

        # Check all batchsize dims have the same letter
        input_dist_role = Role.REPLICA
        bs_chars: Set[str] = set()
        for imeta, in_str in zip(imetas, in_strs):
            if imeta.role.is_distributed:
                bs_chars.add(in_str[0])
                # RuleBase has checked there is only one distributed Role.
                input_dist_role = imeta.role
        V.require(len(bs_chars) <= 1)

        if len(bs_chars) == 1:
            bs_char, = bs_chars
            for imeta, in_str in zip(imetas, in_strs):
                # The letter for batchsize cannot appear in:
                if imeta.role.is_distributed:
                    disallow_sub_in_str = in_strs[1:]
                else:
                    disallow_sub_in_str = in_str
                V.require(bs_char not in disallow_sub_in_str)

            # The output must be a distributed tensor too,
            # and bs_char must be and can only be at position 0
            V.require(out_str[0] == bs_char and bs_char not in out_str[1:])
            out_role = input_dist_role

        else:
            out_role = Role.REPLICA

        out_shape = tuple(d[out_char] for out_char in out_str)

        dtypes = set(imeta.dtype for imeta in imetas)
        V.require(len(dtypes) == 1)  # accepts a unique dtype
        out_dtype, = dtypes

        if len(in_strs) == 1:
            arg0, = tensors
            assert type(arg0) is Node
            imeta0 = V.assert_non_structured(arg0)

            # NOTE PyTorch doc "Tensor Views" does not mention it, but
            # einsum may return a view:
            #
            # For equation that's equal to, for example, transpose,
            # the resultant tensor is a view!
            # But for equation like `ab->a`, the result is a new allocation.
            undeterminable_view_info = \
                imeta0.view_info.derive_new_undetermined_view(arg0)

            # TODO although it may not be as complicated as reshape,
            # for which calculation of strides is needed, we still make it
            # a undeterminable view.
            return EasierTensorMeta(out_shape, out_dtype, out_role,
                                    view_info=undeterminable_view_info)

        return EasierTensorMeta(out_shape, out_dtype, out_role)


metadata_rule_registry[torch.einsum] = EinsumMetadataRule
