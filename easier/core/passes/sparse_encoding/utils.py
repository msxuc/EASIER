# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, \
    Type, Union, Callable, cast
from typing_extensions import TypeAlias


import torch
import torch.overrides
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument, map_arg
from easier.core.passes.tensor_group_partition import \
    ElemPart, ElemPartArangeIdx, ElemPartReorderedArangeIdx, ElemPartSortedIdx

from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.utils import \
    logger, EasierJitException
import easier.core.module as esr

from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, \
    vector_index_of, zipsort_using_order, \
    get_selector_reducer_idx_partition_pair, get_selectors_reducers


def broadcast_elempart(src: int, elempart: ElemPart) -> ElemPart:
    dist_env = get_runtime_dist_env()

    if dist_env.rank == src:
        [idx_desc, hint] = [elempart.idx_desc, elempart.hint]
        dist_env.broadcast_object_list(src, [idx_desc, hint])

        if not isinstance(idx_desc, ElemPartArangeIdx):
            dist_env.broadcast(src, elempart.idx.to(dist_env.comm_device))

        return elempart

    else:
        [idx_desc, hint] = dist_env.broadcast_object_list(src)

        if not isinstance(idx_desc, ElemPartArangeIdx):
            idx = dist_env.broadcast(src, shape=(elempart.lengths[src],), dtype=elempart.idx.dtype).cpu()
        else:
            idx = torch.arange(idx_desc.start, idx_desc.end)

        # Only elempart.lengths is constantly replicated
        return ElemPart(idx_desc, idx, elempart.lengths, hint)

def _isin_sorted(to_find: torch.Tensor, sorted_tests: torch.Tensor):
    """
    Returns the mask.
    """
    assert to_find.ndim == 1
    assert sorted_tests.ndim == 1

    lowerbound_indexes = torch.searchsorted(sorted_tests, to_find)
    lowerbound_values = sorted_tests[lowerbound_indexes]

    return lowerbound_values == to_find

def isin_elempart(gidx: torch.Tensor, elempart: ElemPart) -> torch.Tensor:
    """
    Test if elements of `gidx` are in `elempart.idx`.

    Return:
    -   the mask tensor for elements of `gidx`.
    """
    if isinstance(
        elempart.idx_desc, (ElemPartArangeIdx, ElemPartReorderedArangeIdx)
    ):
        start, end = elempart.idx_desc.start, elempart.idx_desc.end
        return torch.logical_and(start <= gidx, gidx < end)
    
    elif isinstance(elempart.idx_desc, ElemPartSortedIdx):
        return _isin_sorted(gidx, elempart.idx)

    else:
        return torch.isin(gidx, elempart.idx)

def elempart_isin(elempart: ElemPart, gidx: torch.Tensor, gidx_sorted = False) -> torch.Tensor:
    """
    Test if elements of `elempart.idx` are in `gidx`.
    """
    def _get_arange_mask():
        assert isinstance(
            elempart.idx_desc, (ElemPartArangeIdx, ElemPartReorderedArangeIdx)
        )

        start, end = elempart.idx_desc.start, elempart.idx_desc.end
        arange_mask = torch.zeros((end - start,), dtype=torch.bool)

        # gidx contains duplicates
        gidx_mask = torch.logical_and(start <= gidx, gidx < end)
        arange_mask[gidx[gidx_mask] - start] = True

        return arange_mask

    if isinstance(elempart.idx_desc, ElemPartArangeIdx):
        arange_mask = _get_arange_mask()
        return arange_mask
    
    elif isinstance(elempart.idx_desc, ElemPartReorderedArangeIdx):
        arange_mask = _get_arange_mask()
        return arange_mask[elempart.idx - elempart.idx_desc.start]

    else:
        if gidx_sorted:
            return _isin_sorted(elempart.idx, gidx)
        else:
            return torch.isin(elempart.idx, gidx)

def sort_elempart(elempart: ElemPart) -> ElemPart:
    if isinstance(elempart.idx_desc, (ElemPartArangeIdx, ElemPartSortedIdx)):
        return elempart

    if isinstance(elempart.idx_desc, ElemPartReorderedArangeIdx):
        start, end = elempart.idx_desc.start, elempart.idx_desc.end
        idx_desc = ElemPartArangeIdx(start, end)
        sorted_idx = torch.arange(start, end)
    else:
        idx_desc = ElemPartSortedIdx()
        sorted_idx = elempart.idx.sort()[0]

    hint = f'{elempart.hint}:sorted'
    return ElemPart(idx_desc, sorted_idx, elempart.lengths, hint)

def reorder_elempart(elempart: ElemPart, reordered_idx: torch.Tensor) -> ElemPart:
    if isinstance(elempart.idx_desc, (ElemPartArangeIdx, ElemPartReorderedArangeIdx)):
        start, end = elempart.idx_desc.start, elempart.idx_desc.end
        idx_desc = ElemPartReorderedArangeIdx(start, end)
    else:
        idx_desc = None

    hint = f'{elempart.hint}:reordered'
    return ElemPart(idx_desc, reordered_idx, elempart.lengths, hint)
