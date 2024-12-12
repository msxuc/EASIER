# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Dict, Iterable, List, Set, cast, AbstractSet, Union
from typing_extensions import TypeAlias
import pytest
import torch

from torch.fx.graph import Graph
from torch.fx.node import Node

from easier.core.jit import EasierTracer

from easier.core.module import Selector, Reducer, Tensor
from easier.core.passes.metadata_propagation.metadata import \
    INT32, EasierTensorMeta, EasierTensorMeta, \
    Role, ScalarType, ViewType, View
from easier.core.passes.metadata_propagation.utils import \
    Validation as V
from easier.core.passes.tensor_grouping import \
    EasierTensorDef, EasierTensorGroup, get_node_tensor_group
from easier.core.passes.data_dependency_analysis import \
    get_data_dependency_inputs, get_data_dependency_users, \
    KEY__DATA_DEPENDENCY_USERS, KEY__DATA_DEPENDENCY_INPUTS
from easier.core import passes
import easier as esr
from easier.core.passes.utils import FX


def _assert_deps(inputs: Dict[Node, List[Node]], nodes: Iterable[Node]):
    # Only specify the dep_inputs relationship;
    # Re-generate dep_outputs relationship;
    # For each Node, check:
    # - If dep_input dep_output are correct;
    # - consistency of whether it has or not the dep edges.
    inputs_sets = {k: set(v) for k, v in inputs.items()}
    users_sets = {}
    for k, v in inputs.items():
        for arg in v:
            users: Set[Node] = users_sets.setdefault(arg, set())
            users.add(k)

    for n in nodes:
        if n.op == FX.OUTPUT:
            continue

        assert inputs_sets.get(n, set()) == set(get_data_dependency_inputs(n))
        assert users_sets.get(n, set()) == set(get_data_dependency_users(n))


def _get_viewinfo(node: Node) -> View:
    return V.assert_non_structured(node).view_info


def test_data_dependency__none():
    class M(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = Tensor(torch.zeros([55, 3]), mode='partition')

        def forward(self):
            a = self.v * 2
            b = a + 3
            c = a - b
            d = esr.sum(c)

    m = M()
    graph = EasierTracer().trace(m)
    [jm], [graph] = passes.propagate_metadata([m], [graph])
    [jm], [graph] = passes.group_tensors([m], [graph])
    [jm], [graph] = passes.analyze_data_dependency([m], [graph])

    get_v, mul, add, sub, sum, output = graph.nodes

    for n in graph.nodes:
        assert KEY__DATA_DEPENDENCY_INPUTS not in n.meta
        assert KEY__DATA_DEPENDENCY_USERS not in n.meta


def test_data_dependency__two_path_inplace():
    class M(esr.Module):
        def __init__(self):
            super().__init__()
            self.v55 = Tensor(torch.zeros([55, 3]), mode='partition')
            self.r55_22 = Reducer(torch.LongTensor([0] * 55), 22)
            self.v22 = Tensor(torch.zeros([22, 3]), mode='partition')

        def forward(self):
            view22 = self.v22[:]
            self.r55_22(self.v55, out=self.v22)
            a22 = view22 + 2
            self.v22[:] = 1

            aview22 = a22[:]
            self.r55_22(self.v55, out=a22)
            self.v22.add_(aview22)

            self.v22.sub_(view22)

    m = M()
    graph = EasierTracer().trace(m)
    [jm], [graph] = passes.propagate_metadata([m], [graph])
    [jm], [graph] = passes.group_tensors([m], [graph])
    [jm], [graph] = passes.analyze_data_dependency([m], [graph])

    v22, view22, v55, r55_22_v22, a22_add, set1, \
        aview22, r55_22_a22, add_, sub_, output = graph.nodes

    assert _get_viewinfo(v22) == View(ViewType.ALLOCATED, None)
    assert _get_viewinfo(view22) == View(ViewType.DERIVED, v22)
    assert _get_viewinfo(v55) == View(ViewType.ALLOCATED, None)
    assert _get_viewinfo(r55_22_v22) == View(ViewType.DERIVED, v22)
    assert _get_viewinfo(a22_add) == View(ViewType.ALLOCATED, None)
    assert _get_viewinfo(set1) == View(ViewType.DERIVED, v22)
    assert _get_viewinfo(aview22) == View(ViewType.DERIVED, a22_add)
    assert _get_viewinfo(r55_22_a22) == View(ViewType.DERIVED, a22_add)
    assert _get_viewinfo(add_) == View(ViewType.DERIVED, v22)
    assert _get_viewinfo(sub_) == View(ViewType.DERIVED, v22)

    _assert_deps({
        r55_22_v22: [view22],
        a22_add: [r55_22_v22],
        set1: [r55_22_v22, a22_add],
        r55_22_a22: [aview22],
        add_: [set1, r55_22_a22],
        sub_: [add_]
    }, graph.nodes)


def test_data_dependency__undetermined():
    class M(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = Tensor(torch.zeros([55, 3]), mode='partition')

        def forward(self):
            a = self.v[:]
            b = torch.einsum('ab->ab', a)
            c = b[:]
            d = c + 2
            e = d[:]

    m = M()
    graph = EasierTracer().trace(m)
    [jm], [graph] = passes.propagate_metadata([m], [graph])
    [jm], [graph] = passes.group_tensors([m], [graph])
    [jm], [graph] = passes.analyze_data_dependency([m], [graph])

    v, a, b, c, d, e, output = graph.nodes

    assert _get_viewinfo(v) == View(ViewType.ALLOCATED, None)
    assert _get_viewinfo(a) == View(ViewType.DERIVED, v)
    assert _get_viewinfo(b) == View(ViewType.UNDETERMINED, v)
    assert _get_viewinfo(c) == View(ViewType.UNDETERMINED, v)
    assert _get_viewinfo(d) == View(ViewType.ALLOCATED, None)
    assert _get_viewinfo(e) == View(ViewType.DERIVED, d)
