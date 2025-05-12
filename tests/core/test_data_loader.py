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

from easier.core.runtime.data_loader import \
    DataLoaderBase, InMemoryTensorLoader, H5DataLoader, FulledTensorLoader, \
    ArangeTensorLoader

from ..utils import torchrun_singlenode, get_random_str, have_cuda


def get_in_memory_tensor_loader(
    dtype: torch.dtype, device_type: Literal['cpu', 'cuda']
):
    v = torch.arange(17) * 3 + 1
    v = v.to(dtype=dtype, device=device_type)  # e.g. 'cuda' equals 'cuda:0'
    return InMemoryTensorLoader(v)


def get_h5_tensor_loader(dtype: torch.dtype,
                         device_type: Literal['cpu', 'cuda']):
    fn = get_random_str() + ".hdf5"
    dir = os.path.join(tempfile.gettempdir(), "easier", "tests")
    os.makedirs(dir, exist_ok=True)

    v = torch.arange(17) * 3 + 1

    fpath = os.path.join(dir, fn)
    with h5py.File(fpath, 'w') as f:
        f.create_dataset("d", data=v.numpy())

    return H5DataLoader(fpath, "d", dtype=dtype, device=device_type)


def worker__test_load_by_rank(local_rank: int, world_size: int,
                              data_loader_ctor, dtype: torch.dtype,
                              device_type: str):
    dl: DataLoaderBase = data_loader_ctor(dtype, device_type)
    assert dl.dtype == dtype
    assert dl.device.type == device_type
    assert dl.shape == (17,)

    tensor, start, end = dl.partially_load_by_rank()

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert start == local_rank * 8
    assert end == (8 if local_rank == 0 else 17)
    length = end - start
    assert torch.equal(torch.arange(0, length, dtype=dtype)
                       * 3 + 1 + local_rank * 24, tensor)


def worker__test_load_by_index(local_rank: int, world_size: int,
                               data_loader_ctor, dtype: torch.dtype,
                               device_type: str):
    dl: DataLoaderBase = data_loader_ctor(dtype, device_type)
    assert dl.dtype == dtype
    assert dl.device.type == device_type
    assert dl.shape == (17,)

    v = torch.arange(17) * 3 + 1

    torch.manual_seed(2345 + local_rank)
    idx = torch.randint(0, 17, (20,))
    tensor = dl.partially_load_by_index(idx, chunk_size=7)

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert torch.equal(v[idx], tensor)


def worker__test_fully_load(
    local_rank: int, world_size: int,
    data_loader_ctor, dtype: torch.dtype,
    device_type: str, final_device_type: str
):
    dl: DataLoaderBase = data_loader_ctor(dtype, device_type)
    assert dl.dtype == dtype
    assert dl.device.type == device_type
    assert dl.shape == (17,)

    v = torch.arange(17) * 3 + 1
    v = v.to(final_device_type)

    tensor = dl.fully_load(final_device_type, replicated=False)
    assert tensor.dtype == dtype
    assert tensor.device.type == final_device_type
    if local_rank == 0:
        assert torch.equal(v, tensor)
    else:
        assert not torch.equal(v, tensor)

    tensor = dl.fully_load(final_device_type, replicated=True)
    assert tensor.dtype == dtype
    assert tensor.device.type == final_device_type
    assert torch.equal(v, tensor)


@pytest.mark.parametrize('data_loader_ctor',
                         [get_in_memory_tensor_loader, get_h5_tensor_loader])
@pytest.mark.parametrize('dtype',
                         [torch.int64, torch.float64], ids=['i64', 'f64'])
@pytest.mark.parametrize('device_type', [
    'cpu',
    # no device IDs, all workers use cuda:0.
    pytest.param('cuda', marks=have_cuda)
])
class TestDataLoader:

    @pytest.mark.usefixtures('dummy_dist_env')
    def test_load_chunk(self, data_loader_ctor, dtype: torch.dtype,
                        device_type: str):
        # rank-0 only
        dl: DataLoaderBase = data_loader_ctor(dtype, device_type)
        assert dl.dtype == dtype
        assert dl.device.type == device_type
        assert dl.shape == (17,)

        it = dl.partially_load_by_chunk(7)
        chunks = list(it)

        for chunk in chunks:
            assert chunk.dtype == dtype
            assert chunk.device == torch.device('cpu')

        assert torch.equal(torch.arange(0, 7, dtype=dtype) * 3 + 1, chunks[0])
        assert torch.equal(torch.arange(7, 14, dtype=dtype) * 3 + 1, chunks[1])
        assert torch.equal(torch.arange(
            14, 17, dtype=dtype) * 3 + 1, chunks[2])

    def test_load_by_rank(self, data_loader_ctor, dtype: torch.dtype,
                          device_type: str):
        torchrun_singlenode(2, worker__test_load_by_rank,
                            (data_loader_ctor, dtype, device_type))

    def test_load_by_index(self, data_loader_ctor, dtype: torch.dtype,
                           device_type: str):
        torchrun_singlenode(2, worker__test_load_by_index,
                            (data_loader_ctor, dtype, device_type))

    @pytest.mark.parametrize('final_device_type', [
        'cpu',
        pytest.param('cuda', marks=have_cuda)
    ])
    def test_fully_load(self, data_loader_ctor, dtype: torch.dtype,
                        device_type: str, final_device_type: str):
        torchrun_singlenode(
            2, worker__test_fully_load,
            (data_loader_ctor, dtype, device_type, final_device_type)
        )

    @pytest.mark.usefixtures('dummy_dist_env')
    def test_to(self, data_loader_ctor, dtype: torch.dtype, device_type: str):
        dl: DataLoaderBase = data_loader_ctor(dtype, 'cpu')

        if dtype.is_floating_point:
            dl_f16 = dl.to(dtype=torch.float16)
            assert dl is not dl_f16
            assert dl_f16.device.type == 'cpu'
            assert dl_f16.dtype == torch.float16
        else:
            dl_i8 = dl.to(dtype=torch.int8)
            assert dl is not dl_i8
            assert dl_i8.device.type == 'cpu'
            assert dl_i8.dtype == torch.int8  # not changed

        dl_device = dl.to(device=device_type)
        assert dl is not dl_device
        assert dl_device.device.type == device_type
        assert dl_device.dtype == dtype


def worker__test_load_full_by_rank(local_rank: int, world_size: int,
                                   dtype: torch.dtype, device_type: str):
    dl = FulledTensorLoader(
        42, shape=[17, 2], dtype=dtype, device=torch.device(device_type))
    assert dl.dtype == dtype
    assert dl.device.type == device_type
    assert dl.shape == (17, 2)

    tensor, start, end = dl.partially_load_by_rank()

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert start == local_rank * 8
    assert end == (8 if local_rank == 0 else 17)
    length = end - start
    assert torch.equal(torch.full([length, 2], 42, dtype=dtype), tensor)


def worker__test_load_full_by_index(local_rank: int, world_size: int,
                                    dtype: torch.dtype, device_type: str):

    dl = FulledTensorLoader(
        42, shape=[17, 2], dtype=dtype, device=torch.device(device_type))
    assert dl.dtype == dtype
    assert dl.device.type == device_type
    assert dl.shape == (17, 2)

    torch.manual_seed(2345 + local_rank)
    idx = torch.randint(0, 17, (20,))
    tensor = dl.partially_load_by_index(idx, chunk_size=7)

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert torch.equal(torch.full([20, 2], 42, dtype=dtype), tensor)


@pytest.mark.parametrize('dtype',
                         [torch.int64, torch.float64], ids=['i64', 'f64'])
@pytest.mark.parametrize('device_type', [
    'cpu',
    # no device IDs, all workers use cuda:0.
    pytest.param('cuda', marks=have_cuda)
])
class TestFulledLoader:

    @pytest.mark.usefixtures('dummy_dist_env')
    def test_load_chunk(self, dtype: torch.dtype, device_type: str):
        dl = FulledTensorLoader(
            42, shape=[17, 2], dtype=dtype, device=torch.device(device_type))
        assert dl.dtype == dtype
        assert dl.device.type == device_type
        assert dl.shape == (17, 2)

        it = dl.partially_load_by_chunk(7)
        chunks = list(it)

        for chunk in chunks:
            assert chunk.dtype == dtype
            assert chunk.device == torch.device('cpu')

        assert torch.equal(torch.full([7, 2], 42, dtype=dtype), chunks[0])
        assert torch.equal(torch.full([7, 2], 42, dtype=dtype), chunks[1])
        assert torch.equal(torch.full([3, 2], 42, dtype=dtype), chunks[2])

    def test_load_by_rank(self, dtype: torch.dtype, device_type: str):
        torchrun_singlenode(2, worker__test_load_full_by_rank,
                            (dtype, device_type))

    def test_load_by_index(self, dtype: torch.dtype, device_type: str):
        torchrun_singlenode(2, worker__test_load_full_by_index,
                            (dtype, device_type))


def worker__test_load_arange_by_rank(local_rank: int, world_size: int,
                                     dtype: torch.dtype, device_type: str):
    dl = ArangeTensorLoader(0, 34, 2,
                            dtype=dtype, device=torch.device(device_type))
    assert dl.dtype == dtype
    assert dl.device.type == device_type
    assert dl.shape == (17,)

    tensor, start, end = dl.partially_load_by_rank()

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert start == local_rank * 8
    assert end == (8 if local_rank == 0 else 17)
    assert torch.equal(torch.arange(
        start * 2, end * 2, 2, dtype=dtype), tensor)


def worker__test_load_arange_by_index(local_rank: int, world_size: int,
                                      dtype: torch.dtype, device_type: str):

    dl = ArangeTensorLoader(0, 34, 2,
                            dtype=dtype, device=torch.device(device_type))
    assert dl.dtype == dtype
    assert dl.device.type == device_type
    assert dl.shape == (17,)

    torch.manual_seed(2345 + local_rank)
    idx = torch.randint(0, 17, (20,))
    tensor = dl.partially_load_by_index(idx, chunk_size=7)

    assert tensor.dtype == dtype
    assert tensor.device.type == 'cpu'  # by rank always CPU

    assert torch.equal(idx.to(dtype) * 2, tensor)


@pytest.mark.parametrize('dtype',
                         [torch.int64, torch.float64], ids=['i64', 'f64'])
@pytest.mark.parametrize('device_type', [
    'cpu',
    # no device IDs, all workers use cuda:0.
    pytest.param('cuda', marks=have_cuda)
])
class TestArangeLoader:

    @pytest.mark.usefixtures('dummy_dist_env')
    def test_load_chunk(self, dtype: torch.dtype, device_type: str):
        dl = ArangeTensorLoader(0, 34, 2,
                                dtype=dtype, device=torch.device(device_type))
        assert dl.dtype == dtype
        assert dl.device.type == device_type
        assert dl.shape == (17,)

        it = dl.partially_load_by_chunk(7)
        chunks = list(it)

        for chunk in chunks:
            assert chunk.dtype == dtype
            assert chunk.device == torch.device('cpu')

        assert torch.equal(torch.arange(0, 14, 2, dtype=dtype), chunks[0])
        assert torch.equal(torch.arange(14, 28, 2, dtype=dtype), chunks[1])
        assert torch.equal(torch.arange(28, 34, 2, dtype=dtype), chunks[2])

    def test_load_by_rank(self, dtype: torch.dtype, device_type: str):
        torchrun_singlenode(2, worker__test_load_arange_by_rank,
                            (dtype, device_type))

    def test_load_by_index(self, dtype: torch.dtype, device_type: str):
        torchrun_singlenode(2, worker__test_load_arange_by_index,
                            (dtype, device_type))
