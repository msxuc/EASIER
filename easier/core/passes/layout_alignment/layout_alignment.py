# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import \
    Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, cast
from typing_extensions import Literal

import torch
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument
from torch.fx.graph_module import GraphModule
from easier.core.passes.layout_alignment.layout_info import \
    get_node_layout, is_codegen_node, PermuteLayout
from easier.core.passes.layout_alignment.layout_propagation import \
    EasierDistTensor, propagate_layout_info, \
    KEY__GETATTR_MODULENAME

from easier.core.passes.metadata_propagation.metadata import \
    EasierTensorMeta, Role, get_node_meta
from easier.core.passes.metadata_propagation.utils import \
    get_function_variant, Validation as V
from easier.core.utils import \
    FX, fx_normalize_function_variant_into_kwargs, EasierJitException, logger
import easier.core.runtime.modules as _Runtime


def get_permute_dims(ndim: int) -> Tuple[int, ...]:
    perm_dims = tuple(range(1, ndim)) + (0,)
    return perm_dims


def get_permuteback_dims(ndim: int) -> Tuple[int, ...]:
    deperm_dims = (ndim - 1,) + tuple(range(ndim - 1))
    return deperm_dims


class PermuteLayoutRewriterBase:

    normalize_to_kwargs_only: bool = False

    def rewrite(self, *args, **kwargs) -> None:
        """
        A derived class should implement this method to do **in-place**
        fx.Graph rewriting on a per-operation basis.

        The parameters `*args, **kwargs` is to ease the unification of 
        different ways arguments may be passed, e.g. as Python positional
        arguments or keyword arguments.
        """
        raise NotImplementedError("Derived class should implement this")

    def __init__(self, node: Node, root: torch.nn.Module, callee) -> None:
        self.node = node
        self.root = root
        self.graph = self.node.graph

        # - callable `torch.f` for op='call_function'
        # - callable `torch.Tensor.f` for op='call_method'
        # - instance of subtype of `torch.nn.Module` for op='call_module'
        self.callee = callee

    # TODO
    # We may have a phase, separated from rewriting all Nodes targeting the
    # same Module-typed callee, to do Module-specific rewritings.
    # def pre_rewriting_nodes(self):
    #     pass
    # def post_rewriting_nodes(self):
    #     pass
    # To ease clone the callee when it's a Module
    # def alloc_submod_name(self):
    #     if not hasattr(self.root, new_name): return new_name

    def rewrite_ir_inplace(self) -> None:
        def _naive_invoke_rewrite():
            self.rewrite(*self.node.args, **self.node.kwargs)

        if self.node.op == FX.CALL_FUNCTION:

            if self.normalize_to_kwargs_only:
                kwargs: Dict[str, Argument] \
                    = fx_normalize_function_variant_into_kwargs(
                        self.node.target, self.node.args, self.node.kwargs
                )
                self.rewrite(**kwargs)

            else:
                _naive_invoke_rewrite()

        elif self.node.op == FX.CALL_METHOD:

            if self.normalize_to_kwargs_only:
                # Get the corresponding function variant for a method variant
                # operator, e.g. `torch.Tensor.neg` -> `torch.neg`.
                # For those like `torch.Tensor.repeat` which don't have
                # function variants, this will throw.
                func_variant = get_function_variant(self.callee)

                kwargs: Dict[str, Argument] \
                    = fx_normalize_function_variant_into_kwargs(
                        func_variant, self.node.args, self.node.kwargs
                )

                self.rewrite(**kwargs)

            else:
                _naive_invoke_rewrite()

        else:
            _naive_invoke_rewrite()


permute_layout_rewriter_registry: Dict[
    Union[Callable, Type[torch.nn.Module]],
    Union[Literal['same', 'reject'], Type[PermuteLayoutRewriterBase]]
] = {}


def _try_permute_local_reduction_back(local_reduce: Node) -> bool:
    """
    The `local_reduce: Node` has been validated to have PermuteLayout.TRUE
    layout.

    Currently, a reduction involving all vertexes/edges (i.e. on their 0-th
    dimension) in the original EASIER program will be distributed in this way:
    ```
    assert 0 in DIMS
    KD: bool

    s = sum(v, dim=DIMS, keepdim=KD)  # dim and keepdim may be default values
    # distributed into:
    lr_s = sum(v, dim=DIMS, keepdim=True)           # local reduce
    #                  ~~~~          ~~~~
    replica = all_gather_into_tensor(lr_s)
    orig_s = sum(replica, dim=DIMS, keepdim=KD)     # replica reduce
    #                         ~~~~          ~~
    ```

    The result `lr_s` of the _local reduction_ always has the
    same ndim as the input distributed tensor `v`, therefore we can de-permute
    `wlr_s` opposite to how distributed tensors are permuted.

    P.S. operations not supporting these parameters are forbidden, like
    inner product `vec1 @ vec2`.
    """
    nmeta = get_node_meta(local_reduce)
    # TODO is there any torch op that may return both a dist tensor
    # and a replica? Nonetheless it doesn't matter since we will always
    # depermute its tuple-unpacking, .role=REPLICA `getitem` Node.
    if isinstance(nmeta, EasierTensorMeta):
        if nmeta.role == Role.REPLICA:
            with local_reduce.graph.inserting_before(local_reduce.next):
                de_permuted = local_reduce.graph.call_function(
                    torch.permute,
                    (local_reduce, get_permuteback_dims(nmeta.ndim)), {}
                )

                # Permutation will make the worker-local reduction result
                # not contiguous, we insert an explicit `.contiguous()` call
                # to fit communication's requirement on contiguousness.
                contig = local_reduce.graph.call_method(
                    torch.Tensor.contiguous.__name__,
                    (de_permuted,), {}
                )

            local_reduce.replace_all_uses_with(
                contig,
                lambda user: user is not de_permuted)
            return True

    return False


def _get_registered_for(
        key: Union[Callable, Type[torch.nn.Module]]
) -> Union[Literal['same', 'reject'], Type[PermuteLayoutRewriterBase]]:
    registered = permute_layout_rewriter_registry.get(key, None)
    if registered is None:
        raise EasierJitException(
            f"No rewriter for permute layout registered for {key}"
        )
    return registered  # type: ignore


def _rewrite_layout_inplace(root: torch.nn.Module, node: Node) -> None:
    """
    The `node: Node` has been validated to have PermuteLayout.TRUE layout.
    """
    if node.op == FX.CALL_FUNCTION:
        # Node(op='call_function', target=torch.xxx, args=(...))
        callee = cast(Callable, node.target)
        registered = _get_registered_for(callee)

    elif node.op == FX.CALL_METHOD:
        # Node(op='call_method', target="xxx", args=(%input, ...))
        method_variant_func = getattr(torch.Tensor,
                                      cast(str, node.target))
        assert callable(method_variant_func)
        callee = method_variant_func
        registered = _get_registered_for(callee)

    elif node.op == FX.CALL_MODULE:
        submod_path = cast(str, node.target)
        callee = root.get_submodule(submod_path)
        registered = _get_registered_for(type(callee))

    else:
        return

    if registered == 'same':
        return
    elif registered == 'reject':
        raise EasierJitException(
            f'Node\n`{node.format_node()}`\nis not supposed to'
            ' have a associated layout'
        )
    else:
        rewriter_cls: Type[PermuteLayoutRewriterBase] = registered
        rewriter = rewriter_cls(node, root, callee)
        rewriter.rewrite_ir_inplace()


def rewrite_nodes(modules: List[torch.nn.Module],
                  graphs: List[Graph]):
    """
    The policies to rewrite Graph regarding Nodes that have layouts:

    1.  Calls to GraphModule and 
        subsequent `getitem`s for tuple-unpacking GraphModule's outputs.

        These Nodes are intrinsically decided by codegen
        (and marked as _codegen nodes_). No rewriting is needed.

    2.  Nodes with distributed output tensors:

        The layout of the output tensors are directly specified by the layout
        on those Nodes, and their inputs are supposed to have the same layout
        (required by the minimum-data-movement principle, and ensured by the
        layout propagation pass).

        So the rewriting on these Nodes are mainly to change their arguments
        related to dimensions, shapes, etc. to reflect the associated layouts.

        NOTE The output tensors are ALL distributed, but some inputs may be
        replica constants or tensors. Those replicas are always in
        "default torch layout" and may need per-operation basis adjustment too.

    3.  Nodes with replica output tensors:

        These Nodes represent the first-phase, worker-local reductions for
        distributed tensors. They inherit the layouts of the input distributed
        tensors, and those Nodes' arguments need to be changed to reflect
        the layout too.

        And their output tensors need to be de-permuted,
        as all replicas, when used, need to be in "default torch layout".

    P.S. the output properties of policy#2 #3 may appear on _codegen nodes_,
    but we don't rewrite such Nodes.
    """
    for m, g in zip(modules, graphs):
        # The Rewriter may read metadata or layout info of argument Nodes to
        # the target Node, in case of previous Rewriter may have replaced those
        # argument Nodes, we traverse the g.nodes in a reversed order.
        orig_nodes = list(reversed(g.nodes))  # type: ignore
        for node in orig_nodes:
            node: Node

            if is_codegen_node(node):
                continue

            layout = get_node_layout(node)
            if layout != PermuteLayout.TRUE:
                continue

            _rewrite_layout_inplace(m, node)

            # The permuting-back of worker-local replica is separated from
            # rewriting the Node itself.
            _try_permute_local_reduction_back(node)


def _log_permute_disttensor(dt: EasierDistTensor, getattrs: Set[Node]):
    logger.debug(
        f'Permute layout of {dt.__class__.__name__} referred by '
        + ', '.join(
            f'"{n.meta[KEY__GETATTR_MODULENAME]}.{n.target}"'
            for n in getattrs
        ))


def align_layout(modules: List[torch.nn.Module], graphs: List[Graph]):
    disttensor2getattrs = propagate_layout_info(modules, graphs)

    for dt, getattrs in disttensor2getattrs.items():
        typical_node, *_ = getattrs
        if get_node_layout(typical_node) == PermuteLayout.TRUE:
            permute_dims = get_permute_dims(dt.ndim)
            dt.data = torch.permute(dt.data, permute_dims).contiguous()
            dt.layout_permuted = True
            _log_permute_disttensor(dt, getattrs)

    rewrite_nodes(modules, graphs)

    return modules, graphs
