# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing_extensions import Literal
from unittest.mock import patch
import torch
from torch.fx import GraphModule
import pytest

import h5py
import tempfile
import os

from easier.examples.models import Poisson
import easier as esr

from ..utils import mpirun_singlenode, get_random_str, when_ngpus_ge_2


class Model(esr.Module):
    def __init__(self, nf, device='cpu') -> None:
        super().__init__()
        eqn = Poisson(100, device)
        nv = self.nv = eqn.x.shape[0]
        ne = self.ne = eqn.src.shape[0]

        self.reducer = esr.Reducer(eqn.dst, nv)
        self.selector_src = esr.Selector(eqn.src)
        self.selector_dst = esr.Selector(eqn.dst)

        self.vertex_tensor = esr.Tensor(
            torch.randn((nv, nf)).to(device=device), dist='partition')
        self.edge_tensor = esr.Tensor(torch.randn(
            (ne, nf)).to(device=device), dist='partition')
        self.tensor = esr.Tensor(torch.randn(
            (1, nf)).to(device=device), dist='replicate')

    def forward(self):
        dst = self.selector_dst(self.vertex_tensor)
        src = self.selector_src(self.vertex_tensor)
        res = self.reducer(0.5 * (dst + src) - self.edge_tensor + self.tensor)

        self.vertex_tensor[:] = res
        self.edge_tensor[:] = (dst + src) * 0.5
        self.tensor[:] = esr.sum(self.vertex_tensor) / self.nv


class Model2(esr.Module):
    def __init__(self, m1: Model, nf):
        super().__init__()
        nv = m1.nv

        self.reducer = m1.reducer
        self.selector_src = m1.selector_src
        self.selector_dst = esr.Selector(torch.ones_like(m1.selector_src.idx))

        self.tensor1 = m1.vertex_tensor
        self.tensor2 = esr.Tensor(
            torch.randn((nv, nf)).to(device=m1.vertex_tensor.device),
            dist='partition')

    def forward(self):
        dst = self.selector_dst(self.tensor1)
        src = self.selector_src(self.tensor2)
        res = self.reducer(0.5 * (dst + src))

        self.tensor1[:] = res
        self.tensor2[:] = res * 1.5


def worker__test_jitted_dump(local_rank: int, world_size: int,
                             dumpdir: str, device_type: str):
    torch.manual_seed(2345)
    nonejm, = esr.compile([Model(3, 'cpu')], 'none')  # type: ignore
    nonejm: Model
    nonejm()
    nonejm()
    orig_vertex_t2 = nonejm.vertex_tensor.clone()
    orig_edge_t2 = nonejm.edge_tensor.clone()
    orig_replica_t2 = nonejm.tensor.clone()

    torch.manual_seed(2345)
    m = Model(3, 'cpu')
    jm, = esr.compile([m], 'torch')  # type: ignore
    jm()
    jm.dump(dumpdir)

    m = Model(3, 'cpu')
    m.load(dumpdir)
    jm, = esr.compile([m], 'torch')  # type: ignore
    jm: Model
    jm()

    collected_vertex_t2 = jm.vertex_tensor.collect()
    collected_edge_t2 = jm.edge_tensor.collect()
    collected_replica_t2 = jm.tensor.collect()

    torch.testing.assert_close(collected_vertex_t2, orig_vertex_t2)
    torch.testing.assert_close(collected_edge_t2, orig_edge_t2)
    torch.testing.assert_close(collected_replica_t2, orig_replica_t2)


def worker__test_jitted_shared(local_rank: int, world_size: int,
                               dumpdir1: str, dumpdir2: str, device_type: str):

    torch.manual_seed(2345)
    m1 = Model(3, 'cpu')
    m2 = Model2(m1, 3)
    nonejm1, nonejm2 = esr.compile([m1, m2], 'none')  # type: ignore
    nonejm1: Model
    nonejm2: Model2
    nonejm1()
    nonejm2()
    nonejm1()
    nonejm2()
    orig_vertex_t2 = nonejm1.vertex_tensor.clone()
    orig_edge_t2 = nonejm1.edge_tensor.clone()
    orig_replica_t2 = nonejm1.tensor.clone()

    orig_tensor1_t2 = nonejm2.tensor1.clone()
    orig_tensor2_t2 = nonejm2.tensor2.clone()

    torch.manual_seed(2345)
    m1 = Model(3, 'cpu')
    m2 = Model2(m1, 3)
    jm1, jm2 = esr.compile([m1, m2], 'torch')  # type: ignore
    jm1()
    jm2()
    jm1.dump(dumpdir1)
    jm2.dump(dumpdir2)

    m1 = Model(3, 'cpu')
    m2 = Model2(m1, 3)
    m1.load(dumpdir1)
    m2.load(dumpdir2)
    jm1, jm2 = esr.compile([m1, m2], 'torch')  # type: ignore
    jm1: Model
    jm2: Model2

    assert jm1.reducer is jm2.reducer
    assert jm1.selector_src is jm2.selector_src
    assert jm1.selector_dst is not jm2.selector_dst

    assert jm1.vertex_tensor is jm2.tensor1

    assert jm1.vertex_tensor.elempart is jm2.tensor2.elempart

    jm1()
    jm2()

    collected_vertex_t2 = jm1.vertex_tensor.collect()
    collected_edge_t2 = jm1.edge_tensor.collect()
    collected_replica_t2 = jm1.tensor.collect()

    collected_tensor1_t2 = jm2.tensor1.collect()
    collected_tensor2_t2 = jm2.tensor2.collect()

    torch.testing.assert_close(collected_vertex_t2, orig_vertex_t2)
    torch.testing.assert_close(collected_edge_t2, orig_edge_t2)
    torch.testing.assert_close(collected_replica_t2, orig_replica_t2)
    torch.testing.assert_close(collected_tensor1_t2, orig_tensor1_t2)
    torch.testing.assert_close(collected_tensor2_t2, orig_tensor2_t2)


@pytest.mark.parametrize('device_type', [
    'cpu',
    pytest.param('cuda', marks=when_ngpus_ge_2)
])
class TestModuleDump:

    def test_jitted_dump(self, device_type):
        dumpdir = os.path.join(tempfile.gettempdir(), "easier", "tests",
                               get_random_str())
        mpirun_singlenode(2, worker__test_jitted_dump,
                          (dumpdir, device_type,))

    def test_jitted_shared(self, device_type):
        dumpdir1 = os.path.join(tempfile.gettempdir(), "easier", "tests",
                                get_random_str())
        dumpdir2 = os.path.join(tempfile.gettempdir(), "easier", "tests",
                                get_random_str())
        mpirun_singlenode(2, worker__test_jitted_shared,
                          (dumpdir1, dumpdir2, device_type,))
