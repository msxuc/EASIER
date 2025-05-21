# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Iterable, List, Set, cast, Union
import pytest
import torch

from torch.fx.node import Node

from easier.core.jit import EasierTracer

from easier.core.module import Reducer, Tensor
from easier.core.runtime.jit_engine.jit_engine import \
    JitEngine
from easier.core import passes
import easier as esr
from easier.core.passes.utils import FX

def test():
    pass