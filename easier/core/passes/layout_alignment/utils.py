# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import \
    Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, cast
from typing_extensions import Literal

import torch
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument
from easier.core.passes.layout_alignment.layout_alignment import \
    PermuteLayoutRewriterBase, get_permute_dims, get_permuteback_dims, \
    permute_layout_rewriter_registry

from easier.core.passes.metadata_propagation.metadata import \
    EasierTensorMeta, Role, get_node_meta
from easier.core.passes.metadata_propagation.utils import \
    get_function_variant, broadcast_args, Validation as MetaV
from easier.core.utils import FX
import easier.core.module as esr
import easier.core.runtime.modules as _Runtime


def rewrite_replica_shapenorm_and_permuted(
        shapenorm_ndim: int, replica: Node, replica_ndim: int
):
    assert shapenorm_ndim >= replica_ndim, \
        "Target shape-normalized ndim is less than replica ndim," \
        " not available to be shape-normalized"

    g = replica.graph
    with g.inserting_before(replica.next):
        if shapenorm_ndim != replica_ndim:
            paddim = g.call_function(
                operator.getitem,
                (replica, (None,) * (shapenorm_ndim - replica_ndim)), {}
            )
        else:
            paddim = replica

        permuted = g.call_function(
            torch.permute,
            (paddim, get_permute_dims(shapenorm_ndim))
        )

    return permuted


def rewrite_bc_replicas_as_permuted(*args) -> Tuple[Argument, ...]:
    """
    Insert reshape and permute calls on replica Nodes, to preserve the
    broadcasting semantics after permuting the distributed tensors.

    Args:
    -   *args: Can be Node, constant numbers, but cannot be None
    """
    imetas = [MetaV.assert_non_structured(arg) for arg in args]
    bc_meta = broadcast_args(*imetas)
    bc_ndim = bc_meta.ndim

    res: List[Argument] = []
    for imeta, arg in zip(imetas, args):
        if imeta.role == Role.REPLICA and isinstance(arg, Node):
            permuted = rewrite_replica_shapenorm_and_permuted(
                bc_ndim, arg, imeta.ndim)
            res.append(permuted)
        else:
            res.append(arg)

    return tuple(res)


class _DimParamPermuteLayoutRewriterBase(PermuteLayoutRewriterBase):
    normalize_to_kwargs_only = True  # The universal requirement.
    _dim_param_for_input: bool       # A derived class should set this.

    def rewrite(self,
                dim: Optional[Union[int, Sequence[int]]] = None,
                **kwargs) -> None:
        assert dim is not None, f"{self.node.format_node()} not fit"

        is_int = isinstance(dim, int)
        if is_int:
            dim = [dim]

        if self._dim_param_for_input:
            inmetas = [MetaV.assert_non_structured(
                arg) for arg in self.node.all_input_nodes]
            ndims = set(meta.ndim for meta in inmetas)
            MetaV.require(len(ndims) == 1)
            ndim, = ndims
        else:
            outmeta = MetaV.assert_non_structured(self.node)
            ndim = outmeta.ndim

        # Original `dim` parameters 0,   1, ..., n-1 are mapped to
        #                           n-1, 0, ..., n-2,
        # which is the dims into the depermutation calls.
        lookup_dims = get_permuteback_dims(ndim)
        new_dim = tuple(lookup_dims[d] for d in dim)
        if is_int:
            new_dim, = new_dim

        self.node.args = ()
        self.node.kwargs = {**kwargs, 'dim': new_dim}


class DimParamForInputPermuteLayoutRewriter(
        _DimParamPermuteLayoutRewriterBase
):
    """
    The general Node rewriter for ops like `torch.sum, torch.concat`
    """
    _dim_param_for_input = True


class DimParamForOutputPermuteLayoutRewriter(
        _DimParamPermuteLayoutRewriterBase
):
    """
    The general Node rewriter for ops like `torch.stack, torch.unsqueeze`
    """
    _dim_param_for_input = False
