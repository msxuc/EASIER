# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple, Union, cast

import sympy
import torch



def ____get_offset_exactly_nparts__REMOVE_THIS(
    orig_len: int, nparts: int, part: int
) -> Tuple[int, int]:
    """
    A part will have roughly `orig_len // nparts` elements.
    The remaining elements will be put in the last part.

    Please note how the remaining elements are handled. When the remaining
    elements are treated as an individial part, this auxiliary method cannot
    be used.
    """
    per_worker_len = orig_len // nparts

    start = per_worker_len * part

    if part + 1 == nparts:
        end = orig_len
    else:
        end = per_worker_len * (part + 1)

    return start, end


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
    r_region = sympy.Range(region.start, region.stop, region.step)
    r_s = sympy.Range(selection.start, selection.stop, selection.step)
    r_overlap = cast(sympy.Range, r_region.intersect(r_s))

    if len(r_overlap) == 0:
        assert isinstance(r_overlap, sympy.EmptySet)
        return slice(0, 0)

    # sympy.Range.intersect doesn't preserve the direction/sign-of-step,
    # so we need to reverse it if s.step < 0
    if selection.step < 0:
        r_overlap = r_overlap.reversed
    
    return slice(r_overlap.start, r_overlap.stop, r_overlap.step)


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

    s1_len = len(range(s1.start, s1.stop, s1.step))
    s2_len = len(range(s2.start, s2.stop, s2.step))

    ret_start = s1.start + s2.start * s1.step
    ret_step = s2.step * s1.step
    ret_end = ret_start + 





def get_sliced_region(s):
    pass


def get_strides(shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Innermost-major strides.
    P.S. use Tensor.tolist() to get a List[int] of strides.
    """
    r_shp = torch.tensor(shape + (1,), dtype=torch.int64).flip(dims=[0])
    r_strides = torch.cumprod(r_shp, dim=0)
    strides = r_strides[:-1].flip(dims=[0])
    return strides

