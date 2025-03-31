# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import patch
import contextlib
import torch
import pytest

from easier.core.passes.utils import get_selector_reducer_idx_partition


@contextlib.contextmanager
def prepare_zero_length_partition(
    cross_grp_defs  # Tuple[S|R|T, S|R|T]
):
    """
    In a distributed setting, and on these ahead-of-time compilation stages,
    inject zero-length partitions:

    1.  In tensor_group_partition and when preparing the distributed adjmat:

        `get_selector_reducer_idx_partition` is called to get a idx part and
        the range of that part in idx;

        This case stands for a very small but distributed dataset.

        TODO the local part of the adjmat is not injected by this patch,
        because the adjmat is a concat of all TensorGroups and is unlikely to
        be smaller than world size.

    2.  In distpart, the rank-0 METIS returns with one rank having no vertex
        assignment:

        `metis_wrapper` is called;
        TODO in the future this initial assignment may get refined during
        uncoarsening

        This case stands for e.g. boundary conditions and METIS happens to
        partition into a "#" shape rather than a "*" shape.

    3.  In distpart, the overall k-way partition returns with one rank having
        no vertex assignment (this is after uncoarsening)

        `distpart_kway` is called;

        Additional to #2.

    4.  After tensor_group_partition, inject zero ElemParts, and in a
        cross pattern:
        elemparts[grpI][rankJ] = 0, and elemparts[grpI+1][rankJ+1] = 0,
        so that in sparse_encoding we get both sides of both Selector/Reducer
        subprocedures become zero-length.

    Each kind of results above plays as the boundary between stages, so it's
    OK to inject into each of them.
    """
    yield