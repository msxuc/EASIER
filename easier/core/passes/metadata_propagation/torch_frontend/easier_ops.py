# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import List, Sequence, Tuple, Union, cast
from typing_extensions import Literal, TypeAlias

import torch
import torch.overrides
from torch.fx.node import Node
from easier.core.passes.utils import FX, isinst_checker

from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.passes.metadata_propagation.rule_registry import \
    MetadataRuleBase, metadata_rule_registry
from easier.core.passes.metadata_propagation.metadata import \
    BOOL, EasierTensorMeta, Role, View, ViewType, get_meta_from_ir_literal, \
    promote_scalar_types
from easier.core.passes.metadata_propagation.utils import \
    Validation as V, broadcast_and_validate_shapes
from easier.core.utils import EasierJitException

from easier.core.module import Reducer, Selector
import easier.core.module as esr
from easier.core.runtime.modules import \
    all_gather_into_tensor, HaloExchanger


class _IndexesMatchResult:
    def __init__(self) -> None:
        self.shape_split_offsets: Tuple[int, int]
        self.pos_split_offsets: Tuple[int, int]


def _rewrite_ellipsis_indexing_pos(
    node: Node, idxmatch: _IndexesMatchResult,
    pos: Tuple[Union[None, int, slice, tuple, list, Node, 'ellipsis'], ...]
):
    # For concrete meaning see doc in `_calc_sliced_region()` below.
    #
    # Specifically, if no Ellipsis exists:
    # - `shape_autospan_end == len(shape)`
    # - `pos_ellipsis_start == pos_ellipsis_end < len(pos)`

    (shape_autospan_start, shape_autospan_end) = idxmatch.shape_split_offsets
    (pos_ellipsis_start, pos_ellipsis_end) = idxmatch.pos_split_offsets

    if pos_ellipsis_start == pos_ellipsis_end:
        # no Ellipsis at all, don't rewrite
        return
    else:
        colon_slice = slice(None, None, None)
        new_pos = pos[:pos_ellipsis_start] \
            + (colon_slice,) * (shape_autospan_end - shape_autospan_start) \
            + pos[pos_ellipsis_end:]
        node.update_arg(1, new_pos)  # always 1-th arg


def _calc_sliced_region_shape(
    shape: Tuple[int, ...],
    pos: Tuple[Union[None, int, slice, tuple, list, Node, 'ellipsis'], ...],
    idxmatch=_IndexesMatchResult()
) -> Tuple[int, ...]:
    """
    For example:
    ```
    r = torch.tensor([1,2,3])

    A = torch.rand(
        3,         4,  5, 6,  7)
    #   ==________==   ~~~~~  =
    #   consumed_dims         consumed_dims  # before/after ellipsis
    #                  autospan
    #
    #                  |      |
    # shape_split_offset0    shape_split_offset1


    A[  :,  None,  r,  ...,   2:5]
    #   ==__====__==          ===
    #   pos_split             pos_split  # before/after ellipsis
    #                  ===
    #                  Ellipsis
    #
    #                  |      |
    #   pos_split_offset0    pos_split_offset1
    ```

    If no Ellipsis exists, all components are counted as _before_ the ellipsis.

    ```
    B = torch.rand(
        3,         4,  5, 6, 7)
    #   ==________==   ~~~~~~~
    #   consumed_dims_before            # consumed_dims_after=null
    #                  autospan
    #
    #                  |        |EOL
    # shape_split_offset0       shape_split_offset1

    B[  :,  None,  r]
    #   ==________==
    #   pos_split_before                # pos_split_after=null
    #                                   # Ellipsis=null
    #                 |EOL
    #                 pos_split_offset0 & 1
    ```
    """

    nellipsis = pos.count(Ellipsis)
    V.require(nellipsis <= 1)

    ellipsis_consumed = False
    # How many dims in `shape` are consumed by elements of `pos`
    consumed_shape_ndim_before_ellipsis = 0
    consumed_shape_ndim_after_ellipsis = 0
    # How many elements of `pos` appear either before or after `...` in `pos`
    pos_split_len_before_ellipsis = 0
    pos_split_len_after_ellipsis = 0

    for pslice in pos:
        if pslice is None:
            # don't consume an original dim
            if not ellipsis_consumed:
                pos_split_len_before_ellipsis += 1
            else:
                pos_split_len_after_ellipsis += 1

        elif isinstance(pslice, bool):
            # Even static constant-bool indexing is not supported.
            # and isinstance(x, bool) implies isinstance(x, int). But it's more
            # like using None instead of using ints.
            # We need to dedicatedly exclude it.
            raise EasierJitException("Bool index is not supported")

        elif isinstance(pslice, (int, slice, Node)):
            if not ellipsis_consumed:
                consumed_shape_ndim_before_ellipsis += 1
                pos_split_len_before_ellipsis += 1
            else:
                consumed_shape_ndim_after_ellipsis += 1
                pos_split_len_after_ellipsis += 1

        elif isinstance(pslice, (list, tuple)):
            raise NotImplementedError(
                "Consider not using advanced indexing with lists or tuples"
            )

        elif pslice is Ellipsis:
            ellipsis_consumed = True
        else:
            assert False, "unreachable"

    # The comparison is `<=`, because not necessarily all dims in `shape`
    # must be consumed, e.g. `zeros(2,3,4)[1,...,2] = xxx`
    V.require(
        consumed_shape_ndim_before_ellipsis + consumed_shape_ndim_after_ellipsis
        <= len(shape)
    )

    def _fill_resultant_dimlens(
        dimlens: List[int],
        # `pos` is one part of original `pos`, split by Ellipsis `...`
        pos: Tuple[Union[None, int, slice, tuple, list, Node, 'ellipsis'], ...],
        shape_dim_start: int
    ):
        # Here we will increment the "consumed_shape_ndim" again
        # for each individual index object concretely.
        # So we don't need such a parameter.
        consumed_shape_ndim = 0
        for pslice in pos:
            if pslice is None:
                # unsqueeze an extra len=1 dim
                dimlens.append(1)

            elif isinstance(pslice, int):
                consumed_shape_ndim += 1

            elif isinstance(pslice, slice):
                raw_dimlen = shape[shape_dim_start + consumed_shape_ndim]
                # all are positive offsets
                start, stop, step = pslice.indices(raw_dimlen)
                sliced_dimlen = len(range(start, stop, step))
                dimlens.append(sliced_dimlen)
                consumed_shape_ndim += 1

            elif isinstance(pslice, (list, tuple)):
                # TODO this is much complicated than below:
                # ```
                # dimlens.append(len(pslice))
                # consumed_shape_ndim += 1
                # ```
                # as that one indexing list/tuple can be nested and make
                # difference in the resultant ndim.
                assert False, "unreachable"  # raised NotImplementedError above

            elif isinstance(pslice, Node):
                slice_meta = V.assert_non_structured(pslice)

                # NOTE we don't enforce the `pslice` Node must be a direct
                # reference to a `easier.Tensor` replica tensor, i.e.
                # `pslice.op == 'get_attr'`.
                # It's OK as long as `.role==Role.REPLICA`, so possible values
                # could be intermediate results from `easier.Tensor` or
                # `torch.Tensor` constants.
                V.equals(Role.REPLICA, slice_meta.role)
                V.require(slice_meta.dtype.is_integer)

                # Like static constant bool indexing, bool tensor indexing
                # is not allowed either.
                # The index tensor dtype being boolean means that the resultant
                # shape (always flattened to ndim==1 even if the boolean index
                # is n-d) will dynamically determined by runtime values.
                V.require(slice_meta.dtype != BOOL)

                # The replica int tensor to slice may have ndim > 1
                dimlens.extend(slice_meta.shape)
                consumed_shape_ndim += 1

            else:
                # Ellipsis is not expected here.
                assert False, "unreachable"

    resultant_dimlens_before_ellipsis = []
    resultant_dimlens_after_ellipsis = []

    autospan_shape_ndim = \
        len(shape) - consumed_shape_ndim_before_ellipsis \
        - consumed_shape_ndim_after_ellipsis
    shape_autospan_end = \
        consumed_shape_ndim_before_ellipsis + autospan_shape_ndim
    pos_ellipsis_end = pos_split_len_before_ellipsis + nellipsis

    _fill_resultant_dimlens(resultant_dimlens_before_ellipsis,
                            pos[:pos_split_len_before_ellipsis],
                            shape_dim_start=0)
    _fill_resultant_dimlens(resultant_dimlens_after_ellipsis,
                            pos[pos_ellipsis_end:],
                            shape_dim_start=shape_autospan_end)

    autospan_dims = shape[
        consumed_shape_ndim_before_ellipsis:shape_autospan_end]
    out_shp = tuple(resultant_dimlens_before_ellipsis) + autospan_dims \
        + tuple(resultant_dimlens_after_ellipsis)

    idxmatch.shape_split_offsets = \
        (consumed_shape_ndim_before_ellipsis, shape_autospan_end)
    idxmatch.pos_split_offsets = \
        (pos_split_len_before_ellipsis, pos_ellipsis_end)

    return out_shp


def _strip_prefix_dim1s(
    shape: Tuple[int, ...], max_strip: int = -1
) -> Tuple[int, ...]:
    """
    Args:
    -   shape
    -   max_strip: the maximum number of prefix dim 1s can be striped,
            by default strip all.
    """
    for i, d in enumerate(shape):
        if i == max_strip:
            break
        if d != 1:
            break
    else:  # for-else: handler if `for` terminates rather than breaks.
        i = len(shape)

    return shape[i:]


class _GetItemRule(MetadataRuleBase):
    def propagate(self, arg, pos) -> EasierTensorMeta:
        imeta = get_meta_from_ir_literal(arg)
        if not isinstance(imeta, EasierTensorMeta):
            # When input metadata is not an EasierTensorMeta, it means
            # the input node stands for a structure, like:
            # ```
            # U,S,Vh = torch.linalg.svd(...)
            # x = f(U)
            # ```
            # will result in Nodes:
            # ```
            # svd0 := Node[target=svd](...)
            # getitem0 := Node[target=getitem](svd0, 0)  # U
            # f0 := Node[target=f](getitem0)
            # ```
            assert type(imeta) in [tuple, list]
            ometa = imeta[pos]
            assert isinstance(ometa, EasierTensorMeta), \
                "currently we cannot have multiple levels of nested metas"

            # We treat getitem-ing an item out of a tuple as a view.
            assert ometa.view_info == View(ViewType.ALLOCATED, (arg, pos))
            item_derived_view_info = View(ViewType.DERIVED, (arg, pos))

            return EasierTensorMeta(
                shape=ometa.shape, dtype=ometa.dtype, role=ometa.role,
                view_info=item_derived_view_info
            )

        else:
            # Otherwise, it's
            # tensor slicing, possibly with `slice(B,E,S), None`
            return self._propagate_tensor_getitem(arg, pos)

    def _propagate_tensor_getitem(self, arg, pos) -> EasierTensorMeta:
        if type(pos) is not tuple:
            pos = (pos,)

        colon_slice = slice(None, None, None)
        imeta = V.assert_non_structured(arg)
        if imeta.role.is_distributed:
            # Batch dim must not be sliced or unsqueezed.
            # Users can use `M[:, ***]` or `M[..., ***]`
            V.require(pos[0] in [colon_slice, Ellipsis])

        idxmatch = _IndexesMatchResult()
        out_shp = _calc_sliced_region_shape(imeta.shape, pos, idxmatch)

        # rewrite Ellipsis to multiple colons to ease follow passes
        _rewrite_ellipsis_indexing_pos(self.node, idxmatch, pos)

        # no matter how simple an indexing tensor is,
        # even, for example, `torch.arange(10)` or `torch.tensor([1])`,
        # as long as tensors are used as indexes, the result has ViewType.OTHER
        has_tensor_idx = any(map(isinst_checker(Node), pos))
        if has_tensor_idx:
            view_info = View(ViewType.ALLOCATED, None)
        else:
            view_info = imeta.view_info.derive_new_view(arg)

        return EasierTensorMeta(out_shp, imeta.dtype, imeta.role,
                                view_info=view_info)


class _SetItemRule(MetadataRuleBase):
    def propagate(self, arg, pos, v) -> EasierTensorMeta:
        arg = V.must_of(arg, Node)
        # 'get_attr' Node may point to easier.Tensor
        # as well as torch.Tensor constants. But only EASIER parameter tensors
        # will be associated with `setitem` Node, torch.Tensor constants will
        # be evaluated in tracing-time and be always constants during JIT-time.
        V.equals(arg.op, FX.GET_ATTR)

        # tensor slicing, possibly with `slice(B,E,S), None`
        if type(pos) is not tuple:
            pos = (pos,)

        colon_slice = slice(None, None, None)
        imeta = V.assert_non_structured(arg)
        if imeta.role.is_distributed:
            # Batch dim must not be sliced or unsqueezed.
            # Users can use `M[:, ***]` or `M[..., ***]`
            V.require(pos[0] in [colon_slice, Ellipsis])

        vmeta = V.assert_non_structured(v)

        # Validate `v` can be put into the sliced region
        out_dtype = promote_scalar_types(imeta.dtype, vmeta.dtype)
        V.equals(out_dtype, imeta.dtype)

        idxmatch = _IndexesMatchResult()
        region_shp = _calc_sliced_region_shape(imeta.shape, pos, idxmatch)

        # Imagine `a[1,0:2,1:3] = b[5:6,3:5,6:7]` which is a valid setitem,
        # the LHS has empty shape (2,2), and the RHS has shape (1,1,2)
        striped_v_shape = _strip_prefix_dim1s(vmeta.shape)
        bc_shp = broadcast_and_validate_shapes(region_shp, striped_v_shape)
        V.equals(region_shp, bc_shp)

        # rewrite Ellipsis to multiple colons to ease follow passes
        _rewrite_ellipsis_indexing_pos(self.node, idxmatch, pos)

        self.node.meta['easier_is_inplace'] = self.node.args[0]

        # Although, due to the syntax of the setitem, it cannot be used as
        # a Python value, we still infer correct view info for it.
        # Additionally, `operator.setitem` does return a value, but it's None,
        # i.e. FX tracing will break and we won't even get here.
        view_info = imeta.view_info.derive_new_view(arg)

        return EasierTensorMeta(
            shape=imeta.shape, dtype=imeta.dtype, role=imeta.role,
            view_info=view_info)


metadata_rule_registry[operator.getitem] = _GetItemRule
metadata_rule_registry[operator.setitem] = _SetItemRule


class _SelectorRule(MetadataRuleBase):
    # As long as the Selector module `def forward(self, tensor)`
    # and `def propagate(self, tensor)` here share the same name of
    # argument `tensor`, the input FX IR literal can be bound
    # either it's passed positionally or named.
    def propagate(self, tensor) -> EasierTensorMeta:
        callee = cast(Selector, self.callee)
        imeta = V.must_of(
            V.assert_non_structured(tensor), EasierTensorMeta)

        # The resultant `n` is the length of idx array
        n = self.callee.idx.shape[0]
        shp = (n,) + imeta.shape[1:]

        return EasierTensorMeta(shape=shp, dtype=imeta.dtype,
                                role=Role.PARTITION)


class _ReducerRule(MetadataRuleBase):
    # As long as the Reducer module `def forward(self, tensor)`
    # and `def propagate(self, tensor)` here share the same name of
    # argument `tensor`, the input FX IR literal can be bound
    # either it's passed positionally or named.
    def propagate(self, tensor, out=None) -> EasierTensorMeta:
        callee = cast(Reducer, self.callee)
        imeta = V.must_of(
            V.assert_non_structured(tensor), EasierTensorMeta)
        V.equals(imeta.shape[0], callee.idx.shape[0])

        # The resultant `n` is the specified `Reducer.n`
        n = callee.n
        shp = (n,) + imeta.shape[1:]

        if out is None:
            return EasierTensorMeta(
                shape=shp, dtype=imeta.dtype, role=Role.PARTITION)

        else:
            ometa = V.must_of(
                V.assert_non_structured(out), EasierTensorMeta)
            V.equals(shp, ometa.shape)

            self.node.meta['easier_is_inplace'] = out

            promoted_dtype = promote_scalar_types(imeta.dtype, ometa.dtype)
            V.equals(promoted_dtype, ometa.dtype)  # dtype compatible

            out_view_info = ometa.view_info.derive_new_view(out)

            return EasierTensorMeta(
                shape=ometa.shape, dtype=ometa.dtype, role=ometa.role,
                view_info=out_view_info)

    def ensure_batch_size_consistency(
        self, input_dist_metas: List[EasierTensorMeta]
    ):
        # When `out=` argument is provided, it will violate the default
        # batch size consistency.
        # Reducer does not need such a check.
        return


metadata_rule_registry[Selector] = _SelectorRule
metadata_rule_registry[Reducer] = _ReducerRule


class _AllReduceMetadataRule(MetadataRuleBase):
    def propagate(self, tensor, *args, **kwargs) -> EasierTensorMeta:
        imeta = V.assert_non_structured(tensor)
        V.equals(imeta.role, Role.PARTITION)

        shp = (1,) + imeta.shape[1:]
        return EasierTensorMeta(shape=shp, dtype=imeta.dtype, role=Role.REPLICA)


for allreduce_prim in [esr.sum, esr.norm]:
    metadata_rule_registry[allreduce_prim] = _AllReduceMetadataRule


class _HaloExchangerRule(MetadataRuleBase):
    def propagate(self, local) -> EasierTensorMeta:
        imeta = V.assert_non_structured(local)

        callee_haloxchg: HaloExchanger = self.callee
        shape = tuple(callee_haloxchg.chunk_v.shape)

        return EasierTensorMeta(shape=shape, dtype=imeta.dtype,
                                role=Role.PARTITION)
        # NOTE although the resultant TensorMeta of HaloExchanger is marked
        # with Role=PARTITION, it is not a traditional Role.PARTITION tensor,
        # i.e. it has no dedicated TensorGroup etc.


metadata_rule_registry[HaloExchanger] = _HaloExchangerRule


class _AllGatherIntoTensorRule(MetadataRuleBase):
    def propagate(self, send_tensor,
                  form: Literal['concat', 'stack'] = 'concat'
                  ) -> EasierTensorMeta:
        imeta = V.assert_non_structured(send_tensor)

        dist_env = get_runtime_dist_env()
        shape = list(imeta.shape)
        if form == 'concat':
            shape[0] = shape[0] * dist_env.world_size
        else:
            shape.insert(0, dist_env.world_size)

        return EasierTensorMeta(shape=tuple(shape), dtype=imeta.dtype,
                                role=Role.REPLICA)


metadata_rule_registry[all_gather_into_tensor] = _AllGatherIntoTensorRule
