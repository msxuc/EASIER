# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Union
from types import MethodType
import pytest
from unittest.mock import MagicMock, Mock, patch

import torch
from easier.core.passes.dataflow_distribution.tensor_partition import ElemPart
import easier.core.runtime.dist_env as _JitRuntimeDistEnv
from easier.core.runtime.dist_env import DistEnv
from easier.core.runtime.data_loader import InMemoryTensorLoader
from easier.core.module import Selector, Reducer
from easier.core.passes.dataflow_distribution.dist_pass import \
    rewrite_selector_instance, rewrite_reducer_instance, get_cpu_dist_env
from easier.core.passes.utils import \
    get_runtime_dist_env
from tests.utils import assert_tensor_list_equal


def vec(*longs):
    return torch.LongTensor(longs)


# All parameters to activate this fixture
# must use the same name as this function.
@pytest.fixture(scope='function')
def mock_dist_env():
    mock_dist_env = Mock(spec=DistEnv)
    mock_dist_env.world_size = 2    # by default world_size=2
    mock_dist_env.host_rank = 0
    mock_dist_env.comm_device = torch.device('cpu')

    with \
            patch(
                f'{rewrite_reducer_instance.__module__}.{get_cpu_dist_env.__name__}'
            ) as p1:
        p1.return_value = mock_dist_env

        yield mock_dist_env


def _partially_load_idx(module: Union[Selector, Reducer]):
    # In dist_pass we use EasierInterpreter to pick up S/R and load-by-rank,
    # however we don't have fx.graphs to init EasierInterpreter here.
    assert not module.easier_index_ready
    partial_idx, pstart, pend = \
        module.easier_data_loader.partially_load_by_rank()
    module.idx = partial_idx
    module.easier_idx_part_range = (pstart, pend)
    module.easier_index_ready = True


def test_rewrite_selector_instance(mock_dist_env):
    mock_dist_env.world_size = 3

    input_elempart_idxes = [
        vec(0, 1, 2, 3, 4),
        vec(5, 6, 7, 8, 9, 10, 11, 12, 13),
        vec(14, 15, 16, 17)
    ]
    output_elempart_idxes = [
        vec(1, 4, 7, 10),
        vec(0, 3, 6, 9, 11, 12),
        vec(2, 5, 8)
    ]

    full_idx = vec(
        1, 3, 8, 13, 15,
        2, 0, 9, 7,  13,
        16,  11,   0
    )

    # Rows are what's sent by each rank; i.e. columns are what they receive
    # "slices" are subranges of idx-cells
    output_gidx_slices = [
        [vec(1, 4), vec(0, 3), vec(2)],
        [vec(7), vec(6, 9), vec(5, 8)],
        [vec(10), vec(11, 12), vec()],
    ]
    input_gidx_slices = [
        [vec(3, 15), vec(1, 13), vec(8)],
        [vec(9), vec(0, 13), vec(2, 7)],
        [vec(16), vec(11, 0), vec()],
    ]

    # Halo idxes are in the (sorted) order of input elempart
    # halo_gidxes_to_gather = [
    #     [vec(3), vec(9), vec(15, 16)],
    #     [vec(0, 1), vec(11, 13), vec()],
    #     [vec(2), vec(7, 8), vec()],
    # ]
    halo_lidxes_to_gather = [
        [vec(3), vec(4), vec(1, 2)],
        [vec(0, 1), vec(6, 8), vec()],
        [vec(2), vec(2, 3), vec()],
    ]

    def run():
        """
        Do rank-agnostic config and run the function to test
        """
        rank = mock_dist_env.rank
        input_elempart = ElemPart(input_elempart_idxes[rank], [5, 9, 4])
        output_elempart = ElemPart(output_elempart_idxes[rank], [4, 6, 3])

        mock_dist_env.broadcast.reset_mock()

        def _broadcast(src_rank, *args, **kwargs):
            if mock_dist_env.broadcast.call_count <= 3:
                # broadcast 3 times for output_elemparts
                if src_rank == rank:
                    assert torch.equal(args[0], output_elempart.idx)
                return output_elempart_idxes[src_rank]
            else:
                # broadcast 3 times for input_elemparts
                if src_rank == rank:
                    assert torch.equal(args[0], input_elempart.idx)
                return input_elempart_idxes[src_rank]
        mock_dist_env.broadcast.side_effect = _broadcast

        mock_dist_env.all_to_all.reset_mock()

        def _sparse_all2all(tensors):
            c = mock_dist_env.all_to_all.call_count
            if c == 1:
                # input gidx
                assert_tensor_list_equal(tensors, [
                    input_gidx_slices[mock_dist_env.rank][u] for u in range(3)
                ])
                return [
                    input_gidx_slices[w][mock_dist_env.rank] for w in range(3)
                ]

            elif c == 2:
                # output gidx
                assert_tensor_list_equal(tensors, [
                    output_gidx_slices[mock_dist_env.rank][u] for u in range(3)
                ])
                return [
                    output_gidx_slices[w][mock_dist_env.rank] for w in range(3)
                ]
            elif c == 3:
                # halo lidx
                assert_tensor_list_equal(tensors, [
                    halo_lidxes_to_gather[mock_dist_env.rank][u] for u in range(3)
                ])
                return [
                    halo_lidxes_to_gather[w][mock_dist_env.rank] for w in range(3)
                ]
        mock_dist_env.all_to_all.side_effect = _sparse_all2all

        selector = Selector(full_idx)

        # The partition in data loader is likely to change, we fix the partition
        # results for unit testing.
        def _simulate_partial_load_idx(self: InMemoryTensorLoader):
            assert self.tensor.shape[0] == 13
            starts_ends = [(0, 5), (5, 10), (10, 13)]
            (start, end) = starts_ends[rank]
            return self.tensor[start:end], start, end
        selector.easier_data_loader.partially_load_by_rank = \
            MethodType(_simulate_partial_load_idx, selector.easier_data_loader)
        _partially_load_idx(selector)

        rewrite_selector_instance(
            selector, input_elempart, output_elempart)

        assert mock_dist_env.broadcast.call_count == 6
        assert mock_dist_env.all_to_all.call_count == 3

        return selector

    mock_dist_env.rank = 0
    selector = run()
    # look up in chunk: 3 15 9 16
    assert torch.equal(selector.idx, vec(3, 6, 5, 7))
    assert_tensor_list_equal(
        selector.runtime_halos_local_idxes, [
            halo_lidxes_to_gather[u][mock_dist_env.rank] for u in range(3)
        ])

    mock_dist_env.rank = 1
    selector = run()
    # look up in chunk: 1 13 0 13 11 0
    assert torch.equal(selector.idx, vec(1, 10, 0, 10, 8, 0))
    assert_tensor_list_equal(
        selector.runtime_halos_local_idxes, [
            halo_lidxes_to_gather[u][mock_dist_env.rank] for u in range(3)
        ])

    mock_dist_env.rank = 2
    selector = run()
    # look up in chunk: 8 2 7
    assert torch.equal(selector.idx, vec(2, 0, 1))  # no local inputelem concat
    assert_tensor_list_equal(
        selector.runtime_halos_local_idxes, [
            halo_lidxes_to_gather[u][mock_dist_env.rank] for u in range(3)
        ])


def test_rewrite_reducer_instance(mock_dist_env):
    mock_dist_env.world_size = 3

    input_elempart_idxes = [
        vec(0, 1, 2, 3, 4),
        vec(5, 6, 7, 8, 9, 10, 11, 12, 13),
        vec(14, 15, 16, 17)
    ]
    output_elempart_idxes = [
        vec(1, 4, 7, 10),
        vec(0, 3, 6, 9, 11, 12),
        vec(2, 5, 8)
    ]

    full_idx = vec(
        12, 8, 4, 11, 7, 3,
        1, 0, 7, 6, 11, 10,
        1, 1, 1, 6, 6, 6
    )

    # Rows are what's sent by each rank; i.e. columns are what they receive
    # "slices" are subranges of idx-cells
    output_gidx_slices = [
        [vec(4, 7), vec(12, 11, 3), vec(8)],
        [vec(1, 7, 10), vec(0, 6, 11), vec()],
        [vec(1, 1, 1), vec(6, 6, 6), vec()],
    ]
    input_gidx_slices = [
        [vec(2, 4), vec(0, 3, 5), vec(1)],
        [vec(6, 8, 11), vec(7, 9, 10), vec()],
        [vec(12, 13, 14), vec(15, 16, 17), vec()],
    ]

    # Halo idxes are in the (sorted) order of input elempart
    # halo_gidxes_to_gather = [
    #     [vec(2, 4), vec(6, 8, 11, 12, 13), vec(14)],
    #     [vec(0, 3), vec(5, 7, 9, 10), vec(15, 16, 17)],
    #     [vec(1), vec(), vec()],
    # ]
    halo_lidxes_to_gather = [
        [vec(2, 4), vec(1, 3, 6, 7, 8), vec(0)],
        [vec(0, 3), vec(0, 2, 4, 5), vec(1, 2, 3)],
        [vec(1), vec(), vec()],
    ]

    def run():
        """
        Do rank-agnostic config and run the function to test
        """
        rank = mock_dist_env.rank
        input_elempart = ElemPart(input_elempart_idxes[rank], [5, 9, 4])
        output_elempart = ElemPart(output_elempart_idxes[rank], [4, 6, 3])

        mock_dist_env.broadcast.reset_mock()

        def _broadcast(src_rank, *args, **kwargs):
            if mock_dist_env.broadcast.call_count <= 3:
                # broadcast 3 times for output_elemparts
                if src_rank == rank:
                    assert torch.equal(args[0], output_elempart.idx)
                return output_elempart_idxes[src_rank]
            else:
                # broadcast 3 times for input_elemparts
                if src_rank == rank:
                    assert torch.equal(args[0], input_elempart.idx)
                return input_elempart_idxes[src_rank]
        mock_dist_env.broadcast.side_effect = _broadcast

        mock_dist_env.all_to_all.reset_mock()

        def _sparse_all2all(tensors):
            c = mock_dist_env.all_to_all.call_count
            if c == 1:
                # input gidx
                assert_tensor_list_equal(tensors, [
                    input_gidx_slices[mock_dist_env.rank][u] for u in range(3)
                ])
                return [
                    input_gidx_slices[w][mock_dist_env.rank] for w in range(3)
                ]

            elif c == 2:
                # output gidx
                assert_tensor_list_equal(tensors, [
                    output_gidx_slices[mock_dist_env.rank][u] for u in range(3)
                ])
                return [
                    output_gidx_slices[w][mock_dist_env.rank] for w in range(3)
                ]
            elif c == 3:
                # halo lidx
                assert_tensor_list_equal(tensors, [
                    halo_lidxes_to_gather[mock_dist_env.rank][u] for u in range(3)
                ])
                return [
                    halo_lidxes_to_gather[w][mock_dist_env.rank] for w in range(3)
                ]
        mock_dist_env.all_to_all.side_effect = _sparse_all2all

        reducer = Reducer(full_idx, 13)

        # The partition in data loader is likely to change, we fix the partition
        # results for unit testing.
        def _simulate_partial_load_idx(self: InMemoryTensorLoader):
            assert self.tensor.shape[0] == 18
            starts_ends = [(0, 6), (6, 12), (12, 18)]
            (start, end) = starts_ends[rank]
            return self.tensor[start:end], start, end
        reducer.easier_data_loader.partially_load_by_rank = \
            MethodType(_simulate_partial_load_idx, reducer.easier_data_loader)
        _partially_load_idx(reducer)

        rewrite_reducer_instance(reducer, input_elempart, output_elempart)

        assert mock_dist_env.broadcast.call_count == 6
        assert mock_dist_env.all_to_all.call_count == 3

        return reducer

    mock_dist_env.rank = 0
    reducer = run()
    assert torch.equal(reducer.idx, vec(1, 2, 0, 2, 3, 0, 0, 0))
    assert_tensor_list_equal(
        reducer.runtime_halos_local_idxes, [
            halo_lidxes_to_gather[u][mock_dist_env.rank] for u in range(3)
        ])

    mock_dist_env.rank = 1
    reducer = run()
    assert torch.equal(reducer.idx, vec(5, 4, 1, 0, 2, 4, 2, 2, 2))
    assert_tensor_list_equal(
        reducer.runtime_halos_local_idxes, [
            halo_lidxes_to_gather[u][mock_dist_env.rank] for u in range(3)
        ])

    mock_dist_env.rank = 2
    reducer = run()
    assert torch.equal(reducer.idx, vec(2))
    assert_tensor_list_equal(
        reducer.runtime_halos_local_idxes, [
            halo_lidxes_to_gather[u][mock_dist_env.rank] for u in range(3)
        ])


def test_rewrite_reducer_instance__interleaved_elempart(mock_dist_env):
    mock_dist_env.world_size = 3

    input_elempart_idxes = [
        vec(1, 5, 8, 11, 12, 14),
        vec(0, 3, 9, 17),
        vec(2, 4, 6, 7, 10, 13, 15, 16)
    ]
    assert torch.equal(
        torch.concat(input_elempart_idxes).sort()[0],  # type: ignore
        torch.arange(18)
    )
    output_elempart_idxes = [
        vec(1, 4, 7, 10),
        vec(0, 3, 6, 9, 11, 12),
        vec(2, 5, 8)
    ]

    full_idx = vec(
        12, 8, 4, 11, 7, 3,
        1, 0, 7, 6, 11, 10,
        1, 1, 1, 6, 6, 6
    )

    # Rows are what's sent by each rank; i.e. columns are what they receive
    # "slices" are subranges of idx-cells
    output_gidx_slices = [
        [vec(4, 7), vec(12, 11, 3), vec(8)],
        [vec(1, 7, 10), vec(0, 6, 11), vec()],
        [vec(1, 1, 1), vec(6, 6, 6), vec()],
    ]
    input_gidx_slices = [
        [vec(2, 4), vec(0, 3, 5), vec(1)],
        [vec(6, 8, 11), vec(7, 9, 10), vec()],
        [vec(12, 13, 14), vec(15, 16, 17), vec()],
    ]

    # Halo idxes are in the (sorted) order of input elempart
    # halo_gidxes_to_gather = [
    #     [vec(8, 11, 12, 14), vec(), vec(2, 4, 6, 13)],
    #     [vec(5), vec(0, 3, 9, 17), vec(7, 10, 15, 16)],
    #     [vec(1), vec(), vec()],
    # ]
    halo_lidxes_to_gather = [
        [vec(2, 3, 4, 5), vec(), vec(0, 1, 2, 5)],
        [vec(1), vec(0, 1, 2, 3), vec(3, 4, 6, 7)],
        [vec(0), vec(), vec()],
    ]

    def run():
        """
        Do rank-agnostic config and run the function to test
        """
        rank = mock_dist_env.rank
        input_elempart = ElemPart(input_elempart_idxes[rank], [6, 4, 8])
        output_elempart = ElemPart(output_elempart_idxes[rank], [4, 6, 3])

        mock_dist_env.broadcast.reset_mock()

        def _broadcast(src_rank, *args, **kwargs):
            if mock_dist_env.broadcast.call_count <= 3:
                # broadcast 3 times for output_elemparts
                if src_rank == rank:
                    assert torch.equal(args[0], output_elempart.idx)
                return output_elempart_idxes[src_rank]
            else:
                # broadcast 3 times for input_elemparts
                if src_rank == rank:
                    assert torch.equal(args[0], input_elempart.idx)
                return input_elempart_idxes[src_rank]
        mock_dist_env.broadcast.side_effect = _broadcast

        mock_dist_env.all_to_all.reset_mock()

        def _sparse_all2all(tensors):
            c = mock_dist_env.all_to_all.call_count
            if c == 1:
                # input gidx
                assert_tensor_list_equal(tensors, [
                    input_gidx_slices[mock_dist_env.rank][u] for u in range(3)
                ])
                return [
                    input_gidx_slices[w][mock_dist_env.rank] for w in range(3)
                ]

            elif c == 2:
                # output gidx
                assert_tensor_list_equal(tensors, [
                    output_gidx_slices[mock_dist_env.rank][u] for u in range(3)
                ])
                return [
                    output_gidx_slices[w][mock_dist_env.rank] for w in range(3)
                ]
            elif c == 3:
                # halo lidx
                assert_tensor_list_equal(tensors, [
                    halo_lidxes_to_gather[mock_dist_env.rank][u] for u in range(3)
                ])
                return [
                    halo_lidxes_to_gather[w][mock_dist_env.rank] for w in range(3)
                ]
        mock_dist_env.all_to_all.side_effect = _sparse_all2all

        reducer = Reducer(full_idx, 13)

        # The partition in data loader is likely to change, we fix the partition
        # results for unit testing.
        def _simulate_partial_load_idx(self: InMemoryTensorLoader):
            assert self.tensor.shape[0] == 18
            starts_ends = [(0, 6), (6, 12), (12, 18)]
            (start, end) = starts_ends[rank]
            return self.tensor[start:end], start, end
        reducer.easier_data_loader.partially_load_by_rank = \
            MethodType(_simulate_partial_load_idx, reducer.easier_data_loader)
        _partially_load_idx(reducer)

        rewrite_reducer_instance(reducer, input_elempart, output_elempart)

        assert mock_dist_env.broadcast.call_count == 6
        assert mock_dist_env.all_to_all.call_count == 3

        return reducer

    mock_dist_env.rank = 0
    reducer = run()
    assert torch.equal(reducer.idx, vec(2, 3, 0, 0, 1, 2, 0, 0))
    assert_tensor_list_equal(
        reducer.runtime_halos_local_idxes, [
            halo_lidxes_to_gather[u][mock_dist_env.rank] for u in range(3)
        ])

    mock_dist_env.rank = 1
    reducer = run()
    assert torch.equal(reducer.idx, vec(1, 5, 4, 2, 2, 0, 4, 2, 2))
    assert_tensor_list_equal(
        reducer.runtime_halos_local_idxes, [
            halo_lidxes_to_gather[u][mock_dist_env.rank] for u in range(3)
        ])

    mock_dist_env.rank = 2
    reducer = run()
    assert torch.equal(reducer.idx, vec(2))
    assert_tensor_list_equal(
        reducer.runtime_halos_local_idxes, [
            halo_lidxes_to_gather[u][mock_dist_env.rank] for u in range(3)
        ])
