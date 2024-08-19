# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union, cast
from dataclasses import dataclass
from typing_extensions import OrderedDict

import torch
from torch.fx.graph import Graph
from torch.fx.node import Node

import numpy as np
import scipy.sparse
from mgmetis import parmetis
from torch.nn.modules import Module

import easier.core.module as esr
from easier.core.passes.tensor_grouping import \
    EasierTensorGroup, get_node_tensor_group
from easier.core.passes.utils import \
    EasierInterpreter, OrderedSet, \
    normalize_reducer_call_into_args, normalize_selector_call_into_args
from easier.core.runtime.dist_env import \
    get_cpu_dist_env, get_mpi_communicator
from easier.core.utils import EasierJitException, logger


def parallel_partition_graph(
    world_size: int, rank: int,
    subadjmat_height: int, adjmat_width: int, vtxdist: torch.Tensor,
    local_rowids: torch.Tensor, local_colids: torch.Tensor
):
    """
    Args:
    -   vtxdist:
            Specifies how the `indptr` is splitted on workers,
            its values are the offsets to `indptr` vector, e.g. [0, N, 2N, ...]
            where N stands for the length of a slice in `indptr`,
            i.e. the adjmat_height.
    """

    assert local_rowids.ndim == local_colids.ndim == 1
    assert local_rowids.shape == local_rowids.shape

    local_rowids = local_rowids.detach().cpu().numpy()
    local_colids = local_colids.detach().cpu().numpy()

    # The value to construct the CSR matrix doesn't matter,
    # as long as it's not zero.
    csr_data = np.ones((local_rowids.shape[0],))

    graph = scipy.sparse.csr_matrix(
        (csr_data, (local_rowids, local_colids)),
        shape=(subadjmat_height, adjmat_width)
    ).tolil()
    # scipy warns against `setdiag` on CSR. LIL format is recommended instead.

    # ParMETIS require the diagonal (relatively to the global adjmat)
    # are zeros and excluded from sparsity.
    off_diag: int = int(vtxdist[rank])
    graph.setdiag(0, off_diag)
    graph = graph.tocsr()

    comm = get_mpi_communicator()
    # `ncuts` is already summed up and replicated;
    # `local_membership` works like this:
    #   the result of`AllGather(local_membership)` is the result of
    #   non-distributed version of graph partitioning.
    ncuts, local_membership = parmetis.part_kway(
        world_size, graph.indptr, graph.indices, vtxdist, comm)

    local_membership = torch.tensor(local_membership)

    return ncuts, local_membership


@dataclass
class CommPair:
    src_tensor_group: EasierTensorGroup
    src_idx_partition: torch.Tensor  # may be any integer dtype

    dst_tensor_group: EasierTensorGroup
    dst_idx_partition: torch.Tensor

    def get_symmetric_pair(self):
        return CommPair(
            src_tensor_group=self.dst_tensor_group,
            src_idx_partition=self.dst_idx_partition,
            dst_tensor_group=self.src_tensor_group,
            dst_idx_partition=self.src_idx_partition
        )


class CommPairCollector(EasierInterpreter):
    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        super().__init__(modules, graphs)

        self.visited = set()

        # Only Select-ed and Reduce-ed TensorGroups are involved in these:
        self.tensor_group_to_matedge_offsets: \
            OrderedDict[EasierTensorGroup, int] = OrderedDict()
        self.accum_n = 0

        # In IR-reference order, same for all workers.
        # Does not include the symmetric part for the communication operations.
        self.comm_pairs: List[CommPair] = []

    def _init_tensor_group_offset(self, tensor_group: EasierTensorGroup):
        if tensor_group not in self.tensor_group_to_matedge_offsets:
            offset = self.accum_n
            self.tensor_group_to_matedge_offsets[tensor_group] = offset
            self.accum_n += tensor_group.n

    def if_call_module(self, module: Module):
        if module in self.visited:
            # We assume the TensorGroup equivalency based on module instance
            return

        node = self.current_node

        if isinstance(module, esr.Selector):
            assert module.easier_idx_part_range is not None, \
                "Selector.idx must have been partially loaded by rank"
            partial_idx = module.idx
            pstart, pend = module.easier_idx_part_range

            src_idx = partial_idx
            dst_idx = torch.arange(pstart, pend)
            input_node = normalize_selector_call_into_args(
                *node.args, **node.kwargs)

        elif isinstance(module, esr.Reducer):
            assert module.easier_idx_part_range is not None, \
                "Reducer.idx must have been partially loaded by rank"
            partial_idx = module.idx
            pstart, pend = module.easier_idx_part_range

            # 1-by-1 read input, randomly write output
            src_idx = torch.arange(pstart, pend)
            dst_idx = partial_idx
            input_node, _out_node = normalize_reducer_call_into_args(
                *node.args, **node.kwargs)

        else:
            raise EasierJitException(
                f'{type(module)} is not supported to appear in'
                ' tensor grouping'
            )

        self.visited.add(module)
        assert isinstance(input_node, Node)
        # For Reducer, even `out=` parameter is given, the Node itself
        # has metadata as output too.

        src_tensor_group = get_node_tensor_group(input_node)
        dst_tensor_group = get_node_tensor_group(node)
        assert src_tensor_group is not None
        assert dst_tensor_group is not None

        self._init_tensor_group_offset(src_tensor_group)
        self._init_tensor_group_offset(dst_tensor_group)
        self.comm_pairs.append(CommPair(src_tensor_group, src_idx,
                                        dst_tensor_group, dst_idx))


def partition_tensor_groups_with_adjmat(
    tensor_group_to_matedge_offsets: OrderedDict[EasierTensorGroup, int],
    accum_n: int,
    comm_pairs: List[CommPair]
):
    """
    Args:
    -   tensor_group_to_matedge_offsets:
            In the order of how related Selector/Reducer appears in the IR.
    -   comm_pairs:
            Communication patterns appearing in the program.
            For each operation of communication (e.g. Reducer.forward)
            only the original one (i.e. the direction) CommPair instance
            should be passed in, this function will add its symmetric part
            to the adjmat.

    This worker is responsible to fill in adjmat entities in
    rows within `[grp_row_lb, grp_row_ub)`. But the indexes specifying adjmat
    entities may locate in other workers.

    ```
                  |         |
        x x x x-----------D----------    --SubAdjMatSpanLB[y]=0
        x     x E                    )
        x     x   |     F   |        ) worker0
        x x x x     G                )
        | e     y y y y y y y y   B  )   --GroupOffset[y]=4  --GrpRowLB[y,0]=0
    - - | -  -  y  -  -  -  - y -  - ) - --GrpRowLB[y,1]=1   ~~GrpRowUB[y,0]=1
        |     g y |         | y      )
        |       y             y      ) worker1
        |   f   y |         | y     C)
        d       y             y A    )
    - - | -  -  y |-  -  -  | y -  - ) - ~~GrpRowUB[y,1]=6
        |       y y y y y y y y      )
        |                 a     z z z) worker2  ~~GroupOffset[y]+N[y]=12
        |       b |         |   z   z)
        |               c       z z z)
         ~~~~~~~~~|~~~~~~~~~|~~~~~~~~    ~~SubAdjMatSpanUB[y]=3
    ```
    The horizontal and vertical lines in the figure stand for both
    the upper bounds of the previous regions (i.e. exclusive) as well as
    the lower bounds of the next regions (i.e. inclusive).

    For example, `Select([5,0,4], Y)` from Group-y into Group-z will fill in
    matrix entities pointed by `A B C`, across two workers;
    and symmetrically, entities `a b c` will also be filled in.

    While `Reduce([5,0,4,2], X)` from Group-x into Group-y will fill in entities
    `D E F G` and symmetrically `d e f g`.
    """

    comm_pairs = [
        pair
        for unidirect_pair in comm_pairs
        for pair in [unidirect_pair, unidirect_pair.get_symmetric_pair()]
    ]

    dist_env = get_cpu_dist_env()
    world_size = dist_env.world_size
    rank = dist_env.rank
    per_worker_n = (accum_n + world_size - 1) // world_size
    empty_buffer = torch.empty((0,), dtype=torch.int64,
                               device=dist_env.comm_device)

    #
    # Construct local sub adjmat
    # by finding all row and col numbers according to communication
    #

    # TODO currently we collect all data then exchange them between workers,
    # therefore there is expected to be a memory peak. We may need to
    # send and free data in stream to decrease the memory peak.
    rowids_for_commpairs: List[torch.Tensor] = []
    colids_for_commpairs: List[torch.Tensor] = []

    for comm_pair in comm_pairs:
        row_tensor_group = comm_pair.src_tensor_group
        rowgrp_idx = comm_pair.src_idx_partition
        col_tensor_group = comm_pair.dst_tensor_group
        colgrp_idx = comm_pair.dst_idx_partition

        rowgrp_offset = tensor_group_to_matedge_offsets[row_tensor_group]
        next_rowgrp_offset = rowgrp_offset + row_tensor_group.n
        colgrp_offset = tensor_group_to_matedge_offsets[col_tensor_group]

        # Limit the region to search
        sub_adjmat_span_lb = rowgrp_offset // per_worker_n
        sub_adjmat_span_ub = (next_rowgrp_offset - 1) // per_worker_n + 1

        subadjmat_rowids_to_send: List[torch.Tensor] = \
            [empty_buffer] * world_size
        adjmat_colids_to_send: List[torch.Tensor] = [empty_buffer] * world_size

        for w in range(sub_adjmat_span_lb, sub_adjmat_span_ub):
            # Region of rows of adjmat on worker-w
            rowgrp_rowid_lb = max(per_worker_n * w, rowgrp_offset)
            rowgrp_rowid_ub = min(per_worker_n * (w + 1), next_rowgrp_offset)

            # The range of indexes for that region on worker-w
            rowgrp_idx_lb: int = rowgrp_rowid_lb - rowgrp_offset
            rowgrp_idx_ub: int = rowgrp_rowid_ub - rowgrp_offset

            # `idx_pos.dtype == torch.bool`: this is bool-slicing
            idx_pos = torch.logical_and(rowgrp_idx_lb <= rowgrp_idx,
                                        rowgrp_idx < rowgrp_idx_ub)
            rowgrp_idx_slice = rowgrp_idx[idx_pos]
            colgrp_idx_slice = colgrp_idx[idx_pos]

            # Adjust row_grp_idx (of domain `[0,row_group.n)`)
            # into row ids of sub adjmat on worker-w
            subadjmat_rowids = (rowgrp_idx_slice +
                                rowgrp_offset) % per_worker_n
            # But `colgrp_idx` does not need adjustment since adjmat is not
            # partitioned among columns.
            adjmat_colids = colgrp_idx_slice + colgrp_offset

            subadjmat_rowids_to_send[w] = subadjmat_rowids
            adjmat_colids_to_send[w] = adjmat_colids

        subadjmat_rowids_tensors = \
            dist_env.all_to_all(subadjmat_rowids_to_send)
        adjmat_colids_tensors = \
            dist_env.all_to_all(adjmat_colids_to_send)

        rowids_for_commpairs.extend(subadjmat_rowids_tensors)
        colids_for_commpairs.extend(adjmat_colids_tensors)
    # endfor comm_pairs

    #
    # Invoke ParMETIS
    #

    # concat all recieved row/col ids and the result is not ordered or uniqued.
    rowids = torch.concat(rowids_for_commpairs)
    colids = torch.concat(colids_for_commpairs)

    # e.g. [0, N, 2N, ..., min(accum_n, wN)]
    vtxdist = torch.arange(world_size + 1) * per_worker_n
    vtxdist[-1].clamp_(max=accum_n)

    subadjmat_height = int(vtxdist[rank + 1] - vtxdist[rank])

    ncuts, local_membership = parallel_partition_graph(
        world_size, rank,
        subadjmat_height=subadjmat_height, adjmat_width=accum_n,
        vtxdist=vtxdist, local_rowids=rowids, local_colids=colids)

    return local_membership


@dataclass
class ElemPart:
    # Only for this worker. Expected to be ordered
    idx: torch.Tensor
    # All lengths are replicated on all workers
    lengths: List[int]


def synchronize_partition_result(
    tensor_group_to_matedge_offsets: OrderedDict[EasierTensorGroup, int],
    accum_n: int,
    local_membership: torch.Tensor
) -> Dict[EasierTensorGroup, ElemPart]:
    """
    Synchronize ParMETIS-partition results ("elempart") into each TensorGroup.

    Remark:
    Synchronization is needed because partition results of a single
    TensorGroup may be scattered on multiple workers.
    And the synchronized partition data is only about the TensorGroup partition
    specific to current worker.
    """
    #
    # Exchange ParMETIS result
    # to construct partition info of every TensorGroup on every worker
    #
    dist_env = get_cpu_dist_env()
    world_size = dist_env.world_size
    rank = dist_env.rank

    empty_buffer = torch.empty((0,), dtype=torch.int64,
                               device=dist_env.comm_device)

    per_worker_n = (accum_n + world_size - 1) // world_size
    subadjmat_rowid_lb = per_worker_n * rank
    subadjmat_rowid_ub = min(per_worker_n * (rank + 1), accum_n)

    synced_elempart: Dict[EasierTensorGroup, ElemPart] = {}

    # `local_membership` is the partition result for this local sub adjmat,
    # which represents rows within `[rowid_lb, rowid_ub)` of adjmat
    # and might (partially) cover multiple TensorGroups.
    # We need to communicate and gather TensorGroup elements assigned to
    # this worker.
    for tensor_group, offset in tensor_group_to_matedge_offsets.items():
        elempart_to_send: List[torch.Tensor] = [empty_buffer] * world_size
        tensor_group_batch_size = tensor_group.n

        if offset >= subadjmat_rowid_ub \
                or offset + tensor_group_batch_size < subadjmat_rowid_lb:
            # This local sub adjmat doesn't cover this TensorGroup.
            pass  # do nothing

        else:
            # `tensor_group` is (partially) covered by `local_membership`
            # and the overlapped part of `local_membership` is:
            local_membership_begin = \
                max(offset, subadjmat_rowid_lb) - subadjmat_rowid_lb
            local_membership_end = \
                min(offset + tensor_group_batch_size, subadjmat_rowid_ub) \
                - subadjmat_rowid_lb

            # .. and this overlapped part specifies membership
            # for TensorGroup elements within:
            tensor_group_begin = max(0, subadjmat_rowid_lb - offset)
            tensor_group_end = min(tensor_group_batch_size,
                                   subadjmat_rowid_ub - offset)

            assert local_membership_end - local_membership_begin \
                == tensor_group_end - tensor_group_begin

            grp_membership = local_membership[
                local_membership_begin:local_membership_end]

            for w in range(world_size):
                elempart = tensor_group_begin \
                    + torch.argwhere(grp_membership == w).ravel()
                elempart_to_send[w] = elempart

        elempart_recv = dist_env.all_to_all(elempart_to_send)

        elempart = torch.concat(elempart_recv)
        elempart_lengths = dist_env.all_gather_into_tensor(
            torch.tensor([elempart.shape[0]], device=dist_env.comm_device)
        ).tolist()

        assert torch.equal(elempart.unique(sorted=True), elempart.sort()[0])

        synced_elempart[tensor_group] = ElemPart(elempart, elempart_lengths)

    # endfor tensor_groups

    return synced_elempart


def insert_noncomm_elemparts(modules: Sequence[esr.Module],
                             elemparts: Dict[EasierTensorGroup, ElemPart]):
    dist_env = get_cpu_dist_env()
    world_size = dist_env.world_size
    rank = dist_env.rank

    named_dtensor: List[Tuple[str, esr.Tensor, esr.Module]] = []
    for i, root in enumerate(modules):
        for name, p in root.named_parameters(prefix=str(i), recurse=True):
            if isinstance(p, esr.Tensor) and p.is_partition:
                named_dtensor.append((name, p, root))
    named_dtensor.sort(key=lambda tp: tp[0])

    for name, p, root in named_dtensor:
        tensor_group = p.easier_tensor_group

        if tensor_group is None:
            # In TensorGrouping we only set group on distributed esr.Tensors
            # that are referenced by `get_attr` Nodes.
            # Such Tensors may still contain data to use outside of JIT,
            # we give each of them an individual singleton Group.
            logger.warning("Distributed esr.Tensor at "
                           f"{root.__class__.__name__}.{name} is never used"
                           " in easier.Module")
            tensor_group = EasierTensorGroup(OrderedSet([p]), n=p.shape[0])
            p.easier_tensor_group = tensor_group

        # No matter it's not involved in communication or not referenced,
        # give it an evenly partitioned ElemPart.
        if tensor_group not in elemparts:
            n = tensor_group.n

            per_worker_n = n // world_size
            residue = n % world_size

            # given the quantity of residue below:
            # assert 0 <= residue < world_size
            # we simply let the last worker have all these residual elements

            start = per_worker_n * rank
            end = start + per_worker_n
            if rank + 1 == world_size:
                end = n

            lengths = [
                per_worker_n
                for w in range(world_size)
            ]
            lengths[-1] += residue

            elemparts[tensor_group] = ElemPart(
                torch.arange(start, end), lengths)


def partition_tensor_groups(
    modules: List[esr.Module], graphs: List[Graph]
) -> Dict[EasierTensorGroup, ElemPart]:
    comm_pairs_collector = CommPairCollector(modules, graphs)
    comm_pairs_collector.run()

    elemparts: Dict[EasierTensorGroup, ElemPart]
    if len(comm_pairs_collector.comm_pairs) > 0:
        local_membership = partition_tensor_groups_with_adjmat(
            comm_pairs_collector.tensor_group_to_matedge_offsets,
            comm_pairs_collector.accum_n,
            comm_pairs_collector.comm_pairs)

        elemparts = synchronize_partition_result(
            comm_pairs_collector.tensor_group_to_matedge_offsets,
            comm_pairs_collector.accum_n,
            local_membership)  # type: ignore
    else:
        elemparts = {}

    # Partition "orphan" esr.Tensors that are never involved in communication.
    insert_noncomm_elemparts(modules, elemparts)

    return elemparts
