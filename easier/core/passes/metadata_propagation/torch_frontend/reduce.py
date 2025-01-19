# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Sequence, Union, cast

import torch

from easier.core.passes.metadata_propagation import \
    Role, EasierTensorMeta, MetadataRuleBase, metadata_rule_registry, \
    convert_torch_dtype_to_scalar_type
from easier.core.passes.metadata_propagation.utils import \
    Validation, get_method_variant, split_list_by_indexes
from easier.core.utils import EasierJitException

# TODO can be extended to _general reduce_ ops, like `amax, amin` too.


class NonBatchReduceMetadataRule(MetadataRuleBase):
    """
    General reduce rule.
    (Reduction on the batch dimension of vertex tensors is a EASIER primitive.
    Reduction on the batch dimension of edge tensors is disallowed)

    Normally a reduction op has 2 overloads:
    -   f(input, *ARGS, dtype=None)
        Reduce to a .Role=REPLICA and 0-order tensor
    -   f(input, *ARGS, dim, keepdim=False, dtype=None)
        Reduce according to `dim` dimensions.
        `dim` arg can be None or a single int.
    It shows that the argument space of the 2nd overload is a superset of
    the space of the 1st overload, so we can smartly merge the process of them.

    But very importantly, not all reduction ops have compatible callsite (how
    arguments are passed), e.g.
    for `x.sum(d)` and `x.norm(p, d)`, arg `d` is to specify dims, but the
    arg positions are different (the 1st and the 2nd) so the value of the dims
    cannot be extracted positionally for different torch ops.
    Therefore `normalize_to_kwargs_only=True` must be specified,
    and `self.propagate()` must take its arguments all as kwargs.
    """

    normalize_to_kwargs_only = True

    # All arguments are bound via normalized, unordered kwargs dict.
    def propagate(self,
                  # kwargs-only normalized, no positional argument.
                  *,
                  input,
                  dim: Optional[Union[int, Sequence[int]]] = None,
                  keepdim: Optional[bool] = False,
                  dtype: Optional[torch.dtype] = None,
                  **_unused_kwargs,
                  ) -> EasierTensorMeta:
        imeta = Validation.assert_non_structured(input)

        if type(dim) is int:
            dim = [dim]
        elif (dim is None) or \
                (isinstance(dim, Sequence) and len(dim) == 0):
            dim = range(imeta.ndim)
        dim = cast(Sequence[int], dim)

        # TODO unify this logic
        if dtype:
            scalar_type = convert_torch_dtype_to_scalar_type(dtype)
        else:
            scalar_type = imeta.dtype

        # Input tensor may be rank-0 and dim is range(0,0)
        _, left_dimlens, sorted_dims = \
            split_list_by_indexes(imeta.shape, dim)

        if sorted_dims[0] == 0:
            if imeta.role == Role.PARTITION:
                raise EasierJitException(
                    "Use EASIER reduce primitive instead for reduction on"
                    " vertex tensors"
                )

        if keepdim:
            shape = list(imeta.shape)
            for p in sorted_dims:
                shape[p] = 1
        else:
            shape = left_dimlens

        return EasierTensorMeta(
            shape=tuple(shape), dtype=scalar_type, role=imeta.role
        )


for reduceop in [
        torch.sum, torch.Tensor.sum,
        torch.mean, torch.Tensor.mean,
]:
    metadata_rule_registry[reduceop] = NonBatchReduceMetadataRule


"""
NOTE `torch.norm` is an alias to `def torch.functional.norm()` which is a
    PyTorch-Python-layer wrapper, not a direct exposure of a native function
    (whose Python callable is `torch._VF.norm`)

So for function variant `torch.norm`, which is also a normalization target that
method variant `torch.Tensor.norm` gets dispatched to, we have signatures:

# from torch/functional.py
def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
# type: (Tensor, str, Optional[List[int]], bool, Optional[Tensor], Optional[int]
#       ) -> Tensor
def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
# type: (Tensor, Optional[number], Optional[List[int]], bool, Optional[Tensor],
#        Optional[int]) -> Tensor
def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
# type: (Tensor, Optional[number], Optional[int], bool, Optional[Tensor],
#        Optional[int]) -> Tensor
def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
# type: (Tensor, str, Optional[int], bool, Optional[Tensor], Optional[int]
#       ) -> Tensor

# and for linalg.norm from native_functions.yaml
- func: linalg_norm(
    Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False,
    *, ScalarType? dtype=None)
- func: linalg_norm.ord_str(
    Tensor self, str ord, int[1]? dim=None, bool keepdim=False,
    *, ScalarType? dtype=None)
"""
for normop in [torch.linalg.norm, torch.norm, torch.Tensor.norm]:
    metadata_rule_registry[normop] = NonBatchReduceMetadataRule
