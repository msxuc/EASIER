# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
from typing import Dict, Iterable, List, Set, Union, cast
from httpx import patch
import pytest
import torch
from unittest.mock import patch

from torch.fx.node import Node
from torch.fx.graph import Graph

from easier.core.jit import EasierTracer

from easier.core.module import Reducer, Tensor
from easier.core.passes.data_dependency_analysis import \
    get_data_dependency_inputs, get_data_dependency_users, \
    KEY__DATA_DEPENDENCY_USERS, KEY__DATA_DEPENDENCY_INPUTS
from easier.core.runtime.metadata import ViewSrc, get_node_view_src
from easier.core.runtime.jit_engine.jit_engine import JitEngine as _orig_JE
from easier.core import passes
import easier as esr
from easier.core.passes.utils import FX


class JitEngine(_orig_JE):
    """
    After 1st run, do data dependency analysis only.
    """

    def compile_after_first_run(self):
        ms, gs = [self.module], [self.graph]
        ms, gs = passes.analyze_data_dependency(ms, gs)
        [self.module], [self.graph] = ms, gs


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


def _get_viewsrc_node(node: Node) -> Union[Node, None]:
    view_src = get_node_view_src(node)
    if view_src is None:
        return None
    else:
        assert isinstance(view_src, ViewSrc)
        return view_src.node


@pytest.mark.usefixtures('dummy_dist_env')
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
    from easier.core.jit import _fully_load_data_backend_none
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)

    JitEngine(m, graph).forward()

    get_v, mul, add, sub, sum, output = graph.nodes

    for n in graph.nodes:
        assert KEY__DATA_DEPENDENCY_INPUTS not in n.meta
        assert KEY__DATA_DEPENDENCY_USERS not in n.meta


@pytest.mark.usefixtures('dummy_dist_env')
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
    from easier.core.jit import _fully_load_data_backend_none
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    JitEngine(m, graph).forward()

    v22, view22, v55, r55_22_v22, a22_add, set1, \
        aview22, r55_22_a22, add_, sub_, output = graph.nodes

    assert _get_viewsrc_node(v22) == v22
    assert _get_viewsrc_node(view22) == v22
    assert _get_viewsrc_node(v55) == v55
    assert _get_viewsrc_node(r55_22_v22) == v22
    assert _get_viewsrc_node(a22_add) == a22_add
    assert _get_viewsrc_node(set1) == v22
    assert _get_viewsrc_node(aview22) == a22_add
    assert _get_viewsrc_node(r55_22_a22) == a22_add
    assert _get_viewsrc_node(add_) == v22
    assert _get_viewsrc_node(sub_) == v22

    _assert_deps({
        r55_22_v22: [view22],
        a22_add: [r55_22_v22],
        set1: [r55_22_v22, a22_add],
        r55_22_a22: [aview22],
        add_: [set1, r55_22_a22],
        sub_: [add_]
    }, graph.nodes)


@pytest.mark.usefixtures('dummy_dist_env')
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
    from easier.core.jit import _fully_load_data_backend_none
    _fully_load_data_backend_none([m], 'cpu')
    graph = EasierTracer().trace(m)
    JitEngine(m, graph).forward()

    v, a, b, c, d, e, output = graph.nodes

    assert _get_viewsrc_node(v) == v
    assert _get_viewsrc_node(a) == v
    assert _get_viewsrc_node(b) == v
    assert _get_viewsrc_node(c) == v
    assert _get_viewsrc_node(d) == d
    assert _get_viewsrc_node(e) == d


@pytest.mark.usefixtures('dummy_dist_env')
def test_data_dependency_nested_call():
    """
    The same Tensor instance is bound to multiple variables, R/W to all those
    variables share the same dep path.
    """
    v = Tensor(torch.zeros([55, 3]), mode='partition')

    class Inner(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = v

        def forward(self):
            a = self.v[:]
            a[:] += 1

    class Intermediate(esr.Module):
        def __init__(self):
            super().__init__()
            self.r = esr.Tensor(torch.zeros(3, 3), mode='replicate')
            self.inner = Inner()

        def forward(self):
            self.r.fill_(9)
            self.inner()

    class Outer(esr.Module):
        def __init__(self):
            super().__init__()
            self.v = v
            self.intermediate = Intermediate()

        def forward(self):
            a = self.v[:]
            a[:] += 1
            self.intermediate()
            a[:] *= 1

    m = Outer()
    from easier.core.jit import compile
    [jm] = compile([m], 'torch')
    jm.forward()

    graph = cast(JitEngine, jm.forward.__self__).graph
    get_attr, a, \
        a_view, add, set_a, \
        call_intermediate, \
        a_view2, mul, set_a2, output \
        = graph.nodes

    assert _get_viewsrc_node(get_attr) == get_attr
    assert _get_viewsrc_node(a) == get_attr
    assert _get_viewsrc_node(a_view) == get_attr
    assert _get_viewsrc_node(add) == add  # view=a[:]; add=view+1; set(a,add)
    assert _get_viewsrc_node(set_a) == get_attr
    assert _get_viewsrc_node(call_intermediate) == None
    assert _get_viewsrc_node(a_view2) == get_attr
    assert _get_viewsrc_node(mul) == mul
    assert _get_viewsrc_node(set_a2) == get_attr

    _assert_deps({
        set_a: [a_view],
        call_intermediate: [set_a],
        a_view2: [call_intermediate],
        set_a2: [a_view2]
    }, graph.nodes)
