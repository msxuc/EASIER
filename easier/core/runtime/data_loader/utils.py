# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from types import EllipsisType
from typing import Iterator, List, Literal, Optional, Sequence, Tuple, TypeAlias, Union, cast
import h5py
import functools
import copy

import numpy as np
import sympy
import torch

from easier.core.runtime.dist_env import \
    get_default_dist_env, get_runtime_dist_env
from easier.core.runtime.utils import check_collective_equality
from easier.core.utils import EasierJitException


def _get_offset_exactly_nparts(
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



def _get_strides(shape: Tuple[int, ...]):
    """
    Innermost-major strides.
    P.S. use Tensor.tolist() to get a List[int] of strides.
    """
    r_shp = torch.tensor(shape + (1,), dtype=torch.int64).flip(dims=[0])
    r_strides = torch.cumprod(r_shp, dim=0)
    strides = r_strides[:-1].flip(dims=[0])
    return strides

