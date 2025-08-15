# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Sequence, Tuple, Union, cast, TYPE_CHECKING

import sympy
import torch

if TYPE_CHECKING:
    # avoid circular imports
    from easier.core.runtime.data_loader.base import \
        SimpleIndex, GeneralIndex

def range_unpack(s: Union[slice, range, sympy.Range]) -> Tuple[int, int, int]:
    """
    Usage:
    ```
    range(*range_unpack(some_slice))
    ```

    `range` is good at `len(range()), list(range())`;
    `slice` is good at `slice.indices()` to trim lower/upperbound regarding
    the dimension length.

    NOTE Do not do `slice(*range)` directly, as `range.__iter__` generate the
    whole list of its points.
    """
    return s.start, s.stop, s.step

def simplify_indices(
    shape: Tuple[int, ...],
    general_indices: Sequence[GeneralIndex]
) -> Sequence[SimpleIndex]:
    """
    The standard behavior of Python and PyTorch for out-of-range slicing
    is to ignore the out-of-range parts.
    But for int-indexing, we need to check it's in-range.
    """
    ndim = len(shape)
    if len(general_indices) > ndim:
        # TODO such cases may be valid if index None is allowed.
        raise IndexError("Too many indices")

    slices: List[slice] = []
    for i, idx in enumerate(general_indices):
        dimlen = shape[i]
        if isinstance(idx, int):
            norm_idx = idx
            if idx < 0:
                norm_idx = dimlen + idx
            if not (0 <= idx and idx < dimlen):
                raise IndexError(
                    f"Index {idx} is out-of-range for dimension {i}"
                    f" with length {dimlen}"
                )
            slices.append(slice(norm_idx, norm_idx + 1))
        elif isinstance(idx, slice):
            # effectively trim the slice regarding the real length,
            # this is required by and in internal subprocedures
            norm_slice = slice(*idx.indices(dimlen))
            slices.append(norm_slice)
        elif idx is Ellipsis:
            norm_slice = slice(0, dimlen)
            slices.append(norm_slice)
        else:
            # TODO support idx==None
            raise IndexError(f"Unexpect index {idx}")
    
    return slices


def get_overlapping_slice(
    region: Union[slice, range], selection: slice
) -> slice:
    """
    Start/stop of both input slices must be converted to non-negative ints.

    The result will have the same step sign as `selection`,
    i.e. the direction in `region` is ignored.
    """
    assert all(v >= 0 for v in [
        region.start, region.stop, selection.start, selection.stop
    ])
    r_region = sympy.Range(*range_unpack(region))
    r_s = sympy.Range(*range_unpack(selection))
    r_overlap = cast(sympy.Range, r_region.intersect(r_s))

    if len(r_overlap) == 0:
        assert isinstance(r_overlap, sympy.EmptySet)
        return slice(0, 0)

    # sympy.Range.intersect doesn't preserve the direction/sign-of-step,
    # so we need to reverse it if s.step < 0
    if selection.step < 0:
        r_overlap = r_overlap.reversed
    
    return slice(*range_unpack(r_overlap))


def compose_slice(s1: slice, s2: slice) -> slice:
    """
    v[s1][s2] == v[compose_slice(s1, s2)].

    Remarks:
    -   s1, s2 and the result slice must not have negative start/stop
        (but step can be negative)
    -   s1, s2 and the result slice can be out-of-range
        (out-of-range part will be simply ignored by the indexing operation,
        this is the common Python behavior)
    """
    assert all(v >= 0 for v in [
        s1.start, s1.stop, s2.start, s2.stop
    ])

    s2_len = len(range(*range_unpack(s2)))

    ret_start = s1.start + s2.start * s1.step
    ret_step = s2.step * s1.step
    # As abs(ret_step) >> abs(s1.step), the upperbound value -- ret_stop --
    # may be beyond the s1.stop, but it's OK since the upperbound is exclusive.
    ret_stop = ret_start + (ret_step + 1) * s2_len

    ret = slice(ret_start, ret_stop, ret_step)

    assert len(range(*range_unpack(ret))) <= len(range(*range_unpack(s1))), \
        "every internal subprocedure should trim slices to be in-range," \
        " so composed slices should never have extra parts"
    return ret

def get_region_shape(shape: Tuple[int, ...], region: Sequence[slice]) -> Tuple[int, ...]:
    ret = []
    for i, dimlen in enumerate(shape):
        if i < len(region):
            s = region[i]
            size = len(range(*s.indices(dimlen)))
        else:
            size = dimlen
        ret.append(size)
    return tuple(ret)


def get_strides(shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Innermost-major strides.
    P.S. use Tensor.tolist() to get a List[int] of strides.
    """
    r_shp = torch.tensor(shape + (1,), dtype=torch.int64).flip(dims=[0])
    r_strides = torch.cumprod(r_shp, dim=0)
    strides = r_strides[:-1].flip(dims=[0])
    return strides

