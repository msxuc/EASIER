# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import enum
import itertools
import math
import operator
from typing import \
    Callable, Dict, Final, List, Optional, Sequence, Tuple, TypeAlias, \
    Union, cast
import numpy
from typing_extensions import Literal
import more_itertools
import pickle

import torch
from torch import nn
from torch.fx.node import Node
from torch.fx.graph import Graph

import easier as esr
from easier.core import passes
from easier.core import module as _EsrMod
from easier.core.passes.utils import \
    FX, OrderedSet, tree_map, normalize_reducer_call_into_args, \
    get_called_module, get_attr_value
from easier.core.utils import EasierJitException, logger
from easier.core.runtime.dist_env import get_runtime_dist_env
from easier.core.runtime.modules import HaloExchanger
from easier.core.runtime.metadata import \
    RuntimeTensorMeta, StructuredTensorMeta, Role, ViewSrc, \
    get_node_meta, set_node_meta, \
    get_runtime_metadata_from_scalar, collect_meta
from easier.core.runtime.jit_engine.handlers import \
    NodeHandlerBase, PreprocessDecision
from easier.core.runtime.jit_engine.values import \
    JitSkipped, jit_skipped, evaluate_node, RuntimeValue, \
    allgather_meta_for_collective_input, exchange_meta_for_halo_exchanger, \
    get_aggregator_neutral_value


def get_value_runtime_metadata(
    addr2viewsrc: Dict[int, Node], node: Node, role: Role, val,
    *,
    _rec_depth=0  # debug-only
) -> StructuredTensorMeta:
    """
    Recursive call.

    Get a probably nested RuntimeTensorMeta structure for some runtime value.
    Such a value is normally the result of evaluating `self.current_node`
    or jit_skipped.

    If a newly allocated tensor is visited, argument `add2viewsrc` dict will
    be updated.

    Args:
    -   addr2viewsrc:
            normally the keys are tensor.untyped_storage().data_ptr()
    -   node
    -   role:
            the statically inferred role for this Node
    -   val:
            the runtime value, will be recursively inspected if there is a
            nested structure.
    """
    if isinstance(val, torch.Tensor):
        # Get the memory address of the underlying memory
        # Will ignore offsets, e.g. `x[:, 2]` has offset `2 * sizeof()`
        addr: int = val.untyped_storage().data_ptr()
        if addr in addr2viewsrc:
            view_src: ViewSrc = addr2viewsrc[addr]
        else:
            view_src: ViewSrc = node
            addr2viewsrc[addr] = view_src

        tensor_runtime_mete = RuntimeTensorMeta(
            role, tuple(val.shape), val.dtype, view_src=view_src
        )
        return tensor_runtime_mete

    elif val is None:
        # TODO assert _rec_depth == 0  # possible?

        # nested esr.Module calls may return None
        return RuntimeTensorMeta(
            Role.REPLICATED, (), torch.int32, view_src=None
        )

    elif val is jit_skipped:
        assert _rec_depth == 0, "jit_skipped is always not nested"

        return RuntimeTensorMeta(
            # It's a convention that Role.DIST and shape=(0,) are used for
            # skipped Nodes/values,
            # no matter what shape[1:] will be in the runtime.
            #
            # The '0' here should be interpreted as the batch size.
            Role.DISTRIBUTED, (0,), torch.int32, view_src=None
        )

    elif isinstance(val, (tuple, list)):
        n_items = len(val)
        item_metas = []

        if _rec_depth > 0:
            # TODO if any such ops, we need View=(Node, List[int]) for
            # arbitrary nested structures and depths.
            raise EasierJitException(
                "Unexpected nested result with nested depth > 1 from"
                f" {node.format_node()}"
            )

        for i in range(n_items):
            item = val[i]
            item_meta = get_value_runtime_metadata(
                addr2viewsrc, node, role, item,
                _rec_depth=_rec_depth+1
            )

            # assume being only one level nested.
            assert isinstance(item, torch.Tensor)
            assert isinstance(item_meta, RuntimeTensorMeta)

            if item_meta.view_src is node:
                # additionally mark it's allocated by a multi-res operation,
                # give i:int we can only have at most 1 nested level.
                item_meta = RuntimeTensorMeta(
                    role=item_meta.role,
                    shape=item_meta.shape,
                    dtype=item_meta.dtype,
                    view_src=(node, i)
                )

            item_metas.append(item_meta)

        item_metas = type(val)(item_metas)  # tuple or list
        return item_metas

    else:
        # Scalar cases may happen for `t.item()` that returns the scalar
        # Python object for singleton tensors.
        return get_runtime_metadata_from_scalar(val)


def get_meta_role_size(meta: StructuredTensorMeta) -> Tuple[Role, int]:
    """
    If any DIST TensorMeta exists in `meta`, a `(DIST, batch_size)` pair
    is returned.
    If there is no DIST, `(REPLICATED, 0)` is returned.

    Args:
    -   meta: Meta | List[Meta]
            Callers can combine many TensorMeta instances, for example,
            all input StructuredTensorMeta objects to an operation are
            conceptually related.
    """
    replica = (Role.REPLICATED, 0)

    def _get_dist_pair(x: RuntimeTensorMeta):
        if x.role == Role.DISTRIBUTED:
            return (Role.DISTRIBUTED, x.shape[0])
        else:
            return replica

    dist_bases = set(collect_meta(
        meta, _get_dist_pair, sentinel=replica
    ))
    assert len(dist_bases) <= 1

    if len(dist_bases) == 1:
        return dist_bases.pop()
    else:
        return replica


class MetadataValidation(NodeHandlerBase):
    """
    Postprocess-only.

    This validates if the shape/dtype/view_info of immediate results
    are *strictly equal* (current requirement) during all forward() calls. 

    Generally, this should be the last postprocess in runtime.
    """

    def __init__(self) -> None:
        super().__init__()
        self.addr2viewsrc: Dict[int, Node]

    def enter_forward(self):
        self.addr2viewsrc = {}

    def exit_forward(self):
        self.addr2viewsrc.clear()

    def postprocess(self, res: RuntimeValue, args, kwargs) -> RuntimeValue:
        prev_meta = get_node_meta(self.current_node)

        def _get_dist_role(x: RuntimeTensorMeta):
            return x.role
        dist_roles = set(collect_meta(
            prev_meta, _get_dist_role, sentinel=Role.REPLICATED
        ))

        assert len(dist_roles) <= 1
        if len(dist_roles) == 1:
            role = Role.DISTRIBUTED
        else:
            role = Role.REPLICATED

        result_meta = get_value_runtime_metadata(
            self.addr2viewsrc, self.current_node, role, res
        )

        # TODO if we support meta changes on the fly, we can just check
        # if batch sizes change.
        # TODO it's better to tolerate view info changes, e.g. a view Node
        # becomes an allocator Node, as long as it does not break the dep edges
        # of data-dep-analysis. Currently by simply equating two Metas we are
        # enforcing that the view info must be exactly the same.
        if prev_meta != result_meta:
            raise EasierJitException(
                "The properties of the result value of the operation"
                f" '{self.current_node.target}' changes:"
                f" {prev_meta} => {result_meta}"
            )

        return res


class SkipZeroLengthNonHalo(NodeHandlerBase):
    """
    Preprocess-only.

    This decides if the Node can be evaluated:
    -   HaloExchanger is always evaluated;
    -   Inspect the Node meta, if it is DIST and zero-length, skip it;
    -   By default, continue the preprocess pipeline.

    Generally, this should be the first preprocess in runtime.
    """

    def preprocess(self, args, kwargs) -> PreprocessDecision:
        if self.current_node.op == FX.CALL_MODULE:
            submod = get_called_module(self.current_module, self.current_node)
            if isinstance(submod, HaloExchanger):
                return PreprocessDecision.CONTINUE

        meta = get_node_meta(self.current_node)

        def _get_dist_bs(x: RuntimeTensorMeta):
            if x.role == Role.DISTRIBUTED:
                return x.shape[0]
            else:
                return None
        dist_sizes = set(collect_meta(meta, _get_dist_bs, sentinel=None))
        if len(dist_sizes) == 1:
            dist_size = dist_sizes.pop()
            if dist_size == 0:
                return PreprocessDecision.SKIP_EVAL

        return PreprocessDecision.CONTINUE


class FirstRunMetadataPropagation(NodeHandlerBase):
    """
    Inspect, propagate and validate metadata in the first run of a forward().

    Preprocesses:
    -   Calculate basic (role, batch_size) info for the Node's output.

        Although tensor_grouping and EasierTensorGroup has such info,
        we need to propagate it again here,
        since we don't have such info in dump/load scenario,
        and tensor_grouping is not aware of runtime-specific HaloExchanger etc.
        The propagation is relatively simple.

    -   Decide if the Node is skippable (role==dist && batch_size==0)

        If it's skippable, it's skipped immediately;
        Otherwise, leave it for successive Handlers to further examine
        (e.g. FirstRunReducerMetaProp might SKIP_EVAL on its own).

        NOTE to decide if skippable, it requires the output (role, batch_size)
        info, that are calculated by the previous substeps.
        So, unlike we have a dedicated SkipZeroLengthNonHalo for runtime,
        we simply decide the PreprocessDecision for the first run here.

    Postprocesses:
    -   Check if the runtime result value, possibly nested, is consistent
        with the basic (role, batch_size);

        Before the first run (or during ahead-of-time compilation)
        we don't know if a Node returns a nested structure
        (e.g. `maxval, maxpos = torch.max(dim=1)`).

    -   Set the nested RuntimeMeta to the Node;
    -   Including jit_skipped cases.
    """

    def __init__(self):
        super().__init__()

        self._replica_role_size: Final = (Role.REPLICATED, 0)

        # jit_skipped is specifically for Role.DIST && batchsize==0,
        # (in contrast, REPLICA && bs==0 won't be skipped)
        # and jit_skipped is always not nested.
        self._skipped_role_size: Final = (Role.DISTRIBUTED, 0)

        #
        # Context for the whole forward() session
        # =================
        self.addr2viewsrc: Dict[int, Node]

        #
        # Context shared by pre-/post-process
        # =================
        # (role, batch_size)
        # - (Role.DISTRIBUTED, 0) means the Node should be skipped;
        # - For Role.REPLICATED, batch_size is always 0.
        self.expected_role_size: Tuple[Role, int]

    def enter_forward(self):
        self.addr2viewsrc = {}

    def exit_forward(self):
        self.addr2viewsrc.clear()

    def preprocess(self, args, kwargs) -> PreprocessDecision:
        """
        The analysis here is essentially to do one-step, one-Node propagation
        of the (role, batch_size) pair.

        An extra action is to decide SKIP_EVAL or not after we get
        (role, batch_size) for the first run.

        Remarks:
        -   This FirstRunMetaProp returns CONTINUE for
            HaloExchangers and Reducers.

        -   At each step, we ensure the JIT time RuntimeTensorMeta is
            consistent with AOT shape/dtype etc.
        """
        should_continue = False

        if self.current_node.op == FX.GET_ATTR:
            tensor = get_attr_value(self.current_module, self.current_node)
            if isinstance(tensor, _EsrMod.Tensor) and tensor.is_partition:
                self.expected_role_size = (Role.DISTRIBUTED, tensor.shape[0])

            else:
                self.expected_role_size = self._replica_role_size

        elif self.current_node.op == FX.CALL_MODULE:
            submod = get_called_module(self.current_module, self.current_node)
            if isinstance(submod, _EsrMod.Module):
                self.expected_role_size = self._replica_role_size

            elif isinstance(submod, _EsrMod.Selector):
                idx_len = submod.idx.shape[0]
                self.expected_role_size = (Role.DISTRIBUTED, idx_len)

            elif isinstance(submod, _EsrMod.Reducer):
                # NOTE for the case `Reducer.foward(halos_concat, out)`
                # the two input metas are not the same, therefore we'll have
                # `len(in_metas) == 2` and out's meta is the resultant meta.
                self.expected_role_size = (Role.DISTRIBUTED, submod.n)

                # let FirstRunReducerMetaProp to decide skip or not
                should_continue = True

            elif isinstance(submod, HaloExchanger):
                concat_len = submod.output_batch_size
                self.expected_role_size = (Role.DISTRIBUTED, concat_len)

                # must eval for the first run
                # even HaloExchange output is zero-length.
                should_continue = True

            else:
                assert False, 'in the first run there will be only 4 cases'

        elif self.current_node.op == FX.OUTPUT:
            self.expected_role_size = self._replica_role_size

        elif self.current_node.target in _EsrMod.easier_aggregators:
            self.expected_role_size = self._replica_role_size

        else:  # call_method or call_function, mapped ops
            arg_metas = list(
                get_node_meta(n) for n in self.current_node.all_input_nodes
            )
            self.expected_role_size = get_meta_role_size(arg_metas)

        if self.expected_role_size == self._skipped_role_size \
                and not should_continue:
            return PreprocessDecision.SKIP_EVAL
        else:
            return PreprocessDecision.CONTINUE

    def postprocess(self, res: RuntimeValue, args, kwargs) -> RuntimeValue:
        """
        Validate Role and batch size consistency between the inferred info
        and runtime, possibly nested, data.
        """
        role = self.expected_role_size[0]
        runtime_meta = get_value_runtime_metadata(
            self.addr2viewsrc, self.current_node, role, res
        )

        def _collect_dist_bs(x: RuntimeTensorMeta):
            if x.role == Role.DISTRIBUTED:
                return x.shape[0]  # including skipped (DIST, (0,), i32, None)
            else:
                return None
        dist_batch_sizes = set(collect_meta(
            runtime_meta, _collect_dist_bs, sentinel=None
        ))

        if len(dist_batch_sizes) == 0:
            if self.expected_role_size != self._replica_role_size:
                raise EasierJitException(
                    f"Expect replicated values but get distributed tensors"
                    f" of batch size {self.expected_role_size[1]}"
                )

        else:
            if len(dist_batch_sizes) != 1:
                raise EasierJitException(
                    f"Distributed multiple-result operation"
                    f" '{self.current_node.format_node()}'"
                    f" does not have a unique batch size, but there are"
                    f" {list(dist_batch_sizes)}"
                )

            dbs = dist_batch_sizes.pop()
            if self.expected_role_size != (Role.DISTRIBUTED, dbs):
                raise EasierJitException(
                    f"Distributed operation"
                    f" '{self.current_node.format_node()}'"
                    f" result has the batch size {dbs},"
                    f" but {self.expected_role_size[1]} is expected"
                )

        set_node_meta(self.current_node, runtime_meta)
        return res


class FirstRunAggregatorMetadataPropagation(NodeHandlerBase):
    """
    Specific propagation for metadata of EASIER aggregators
    in addition to the general FirstRunMetadataPropagation.
    """

    def preprocess_call_function(
        self, function, args: List[RuntimeValue], kwargs
    ) -> PreprocessDecision:
        if function in _EsrMod.easier_aggregators:
            # Set resultant runtime tensor metadata in advance.
            # If on some ranks the input Node is skipped, AggregatorInjector
            # will use that resultant runtime metadata to create a neutral
            # input.
            subshape, dtype = allgather_meta_for_collective_input(args[0])
            set_node_meta(
                self.current_node,
                RuntimeTensorMeta(
                    Role.REPLICATED, (1,) + subshape, dtype, self.current_node
                )
            )

        return PreprocessDecision.CONTINUE


class FirstRunHaloExchangerMetadataPropagation(NodeHandlerBase):
    """
    Specific propagation for metadata of HaloExchanger
    in addition to the general FirstRunMetadataPropagation.
    """

    def preprocess_call_module(
        self, submod, args: List[RuntimeValue], kwargs
    ) -> PreprocessDecision:
        if isinstance(submod, HaloExchanger):

            subshape, dtype = exchange_meta_for_halo_exchanger(
                submod, args[0]
            )
            submod.prepare_buffers(subshape, dtype)

        return PreprocessDecision.CONTINUE


class FirstRunReducerMetadataPropagation(NodeHandlerBase):
    """
    Specific propagation for metadata of Reducer
    in addition to the general FirstRunMetadataPropagation.

    On the first run, some Reducers may lack runtime info,
    because some Reducers have their inputs skipped, but should allocate
    zero-ed output.

    For such Reducers, we need to gather input info from other ranks.
    This is done and can be done in a collective way by visiting on all ranks.
    """

    def preprocess_call_module(
        self, submod, args: List[RuntimeValue], kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        if isinstance(submod, _EsrMod.Reducer):

            subshape, dtype = allgather_meta_for_collective_input(args[0])

            input_val, opt_out_val = normalize_reducer_call_into_args(
                *args, **kwargs
            )
            input_node, opt_out_node = normalize_reducer_call_into_args(
                *self.current_node.args, **self.current_node.kwargs
            )

            view_src = self.current_node
            if isinstance(input_val, JitSkipped):
                if opt_out_node is not None:
                    view_src = \
                        get_node_meta(opt_out_node).view_src  # type: ignore

            set_node_meta(
                self.current_node,
                RuntimeTensorMeta(
                    Role.DISTRIBUTED, (submod.n,) + subshape, dtype, view_src
                )
            )

            if submod.n == 0:
                # In the first run, to allgather Reducer info we need to
                # collectively do this Setup.
                # However, just like FirstRunMetaProp decides skip or not,
                # if this local Reducer is a no-op,
                # after meta prop, we should skip eval.
                return PreprocessDecision.SKIP_EVAL

        return PreprocessDecision.CONTINUE


class AggregatorNeutralInputPreparation(NodeHandlerBase):
    def preprocess_call_function(
        self,
        function,
        args: List[RuntimeValue],
        kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        if function in _EsrMod.easier_aggregators:
            if isinstance(args[0], JitSkipped):
                dist_env = get_runtime_dist_env()
                result_runtime_meta = get_node_meta(
                    self.current_node
                )
                assert isinstance(result_runtime_meta, RuntimeTensorMeta)
                assert result_runtime_meta.role == Role.REPLICATED

                vneutral = get_aggregator_neutral_value(
                    self.current_node.target, result_runtime_meta.dtype
                )

                # TODO We may consider to create the neutral value
                # (per node, also per run if k-dim shape can change)
                # only once.
                # Maybe we can store such values for skipped inputs in
                # JitEngine (also HaloExchanger.zero_length_input can be
                # unified in this way).
                # However given that previous Nodes are skipped, it may have
                # saved enough time to create on the fly.
                arg_neutral = torch.full(
                    result_runtime_meta.shape,
                    fill_value=vneutral,
                    dtype=result_runtime_meta.dtype,
                    device=dist_env.comm_device
                )
                args[0] = arg_neutral

                return PreprocessDecision.GOTO_EVAL

        return PreprocessDecision.CONTINUE


class HaloExchangerZeroLengthInputPreparation(NodeHandlerBase):
    def preprocess_call_module(
        self, submod, args: List[RuntimeValue], kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        if isinstance(submod, HaloExchanger):
            if isinstance(args[0], JitSkipped):

                args[0] = submod.zero_length_input

                return PreprocessDecision.GOTO_EVAL

        return PreprocessDecision.CONTINUE


class NotFullReducerOutputAllocation(NodeHandlerBase):
    """
    Preprocess and postprocess.

    If a local Reducer's input value is jit_skipped, this postprocess()
    will allocate a properly-shaped zero tensor,
    or return the `out` argument if it's specified.

    NOTE
    If the overall output of the local Reducer is zero-length,
    this Handler will not be run, this will be skipped because of
    either SkipZeroLengthNonHalo for runtime or FirstRunReducerMetaProp.
    """

    def preprocess_call_module(
        self, submod, args: List[RuntimeValue], kwargs: Dict[str, RuntimeValue]
    ) -> PreprocessDecision:
        if isinstance(submod, _EsrMod.Reducer):
            input_val, opt_out_val = normalize_reducer_call_into_args(
                *args, **kwargs
            )
            if isinstance(input_val, JitSkipped):

                self.input_val = input_val
                self.opt_out_val = opt_out_val

                return PreprocessDecision.SKIP_EVAL

        return PreprocessDecision.CONTINUE

    def postprocess(self, res: RuntimeValue, args, kwargs) -> RuntimeValue:
        if self.preprocess_decision == PreprocessDecision.SKIP_EVAL:
            assert isinstance(res, JitSkipped)

            input_val, opt_out_val = self.input_val, self.opt_out_val
            del self.input_val
            del self.opt_out_val

            if opt_out_val is None:
                return self._alloc()
            else:
                return opt_out_val

        return res

    def _alloc(self):
        # Pre-Reducer HaloExchanger won't be skipped;
        # Empty input ElemPart won't have reordering Selector.
        dist_env = get_runtime_dist_env()
        result_runtime_meta = get_node_meta(
            self.current_node
        )
        assert isinstance(result_runtime_meta, RuntimeTensorMeta)
        assert result_runtime_meta.role == Role.DISTRIBUTED
        assert result_runtime_meta.view_src == self.current_node

        # Same as Reducer.forward()
        out = torch.zeros(
            result_runtime_meta.shape,
            dtype=result_runtime_meta.dtype,
            device=dist_env.comm_device
        )

        return out


class JitEngine:
    def __init__(self, module: esr.Module, graph: Graph) -> None:
        self.module = module
        self.graph = graph

        self.run_count = 0

        self.first_run_handlers: List[NodeHandlerBase] = [
            FirstRunMetadataPropagation(),

            FirstRunAggregatorMetadataPropagation(),
            FirstRunHaloExchangerMetadataPropagation(),
            FirstRunReducerMetadataPropagation(),

            AggregatorNeutralInputPreparation(),
            HaloExchangerZeroLengthInputPreparation(),
            NotFullReducerOutputAllocation(),
        ]

        self.runtime_handlers: List[NodeHandlerBase] = [
            MetadataValidation(),
            SkipZeroLengthNonHalo(),

            AggregatorNeutralInputPreparation(),
            HaloExchangerZeroLengthInputPreparation(),
            NotFullReducerOutputAllocation(),
        ]

    def _update_handlers(self, handlers: List[NodeHandlerBase]):
        """
        Bind Handlers with latest instances of self.module/graph -- in case
        the instances are changed by the JIT passes.
        """
        for h in handlers:
            h.current_module = self.module
            h.current_graph = self.graph

    def forward(self):
        """
        Run registered Handlers before and after evaluating each Node:
        -   preprocess() are run in sequence,
            postprecess() are run in a reversed order.

        -   Handler.preprocess() can inspect and modify the
            input args/kwargs collection;

        -   Handler.postprocess() can inspect and modify the
            node evaluation result.
        """
        if self.run_count == 0:
            self._update_handlers(self.first_run_handlers)
            handlers = self.first_run_handlers
        else:
            handlers = self.runtime_handlers

        stackframe: Dict[Node, RuntimeValue] = {}

        def _eval(x):
            # instead sf.get(x,x) we want to also check type consistency.
            if isinstance(x, Node):
                return stackframe[x]
            else:
                return x

        for h in handlers:
            h.enter_forward()

        for node in list(self.graph.nodes):
            # args and kwargs collections are mutable for Handlers to modify.
            args = cast(
                List[RuntimeValue],
                list(tree_map(v, _eval) for v in node.args)
            )
            kwargs = cast(
                Dict[str, RuntimeValue],
                {k: tree_map(v, _eval) for k, v in node.kwargs.items()}
            )

            # rank = get_runtime_dist_env().rank
            # s = "    " * rank + "||"
            # logger.info(s + node.format_node())

            for i_handler, handler in enumerate(handlers):
                handler.current_node = node
                decision = handler.preprocess(args, kwargs)
                assert isinstance(decision, PreprocessDecision)
                handler.preprocess_decision = decision

                if decision == PreprocessDecision.SKIP_EVAL:
                    can_eval = False
                    break
                if decision == PreprocessDecision.GOTO_EVAL:
                    can_eval = True
                    break
            else:
                # If no handler breaks/asks to skip eval, we do eval
                can_eval = True

            if can_eval:
                # Only if none of the preprocess steps breaks
                # can we eval the Node;
                # Otherwise it means the Node should be skipped.
                res = evaluate_node(self.module, node, args, kwargs)
            else:
                res = jit_skipped

            for rev_i in range(i_handler, -1, -1):
                rev_handler = handlers[rev_i]
                # Pass in the final args/kwargs values
                res = rev_handler.postprocess(res, args, kwargs)

            stackframe[node] = res

        for h in handlers:
            h.exit_forward()

        if self.run_count == 0:
            ms, gs = [self.module], [self.graph]

            ms, gs = passes.analyze_data_dependency(ms, gs)
            # ms, gs = passes.fuse(ms, gs)
            # ms, gs = passes.codegen(ms, gs)

            [self.module], [self.graph] = ms, gs
            self._update_handlers(self.runtime_handlers)

        self.run_count += 1

        # JitEngine.forward() serves as esr.Module.forward(), and is required
        # to return None.
        return None
