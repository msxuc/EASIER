# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, List, Optional, Sequence, Set, Tuple, cast

import torch
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument
from torch.fx.graph_module import GraphModule

from easier.core.passes.metadata_propagation.metadata import \
    EasierTensorMeta, Role, StructuredTensorMeta, get_node_meta
from easier.core.passes.metadata_propagation.utils import \
    Validation as MetaV
from easier.core.passes.layout_alignment.layout_info import \
    set_codegen_io_layout
from easier.core.passes.layout_alignment.utils import \
    get_permute_dims, get_permuteback_dims
from easier.core.utils import FX, isinst_checker, tree_map


CodegenPermuteLayout = bool
CODEGEN_PERMUTE_TRUE = True
CODEGEN_PERMUTE_FALSE = False


def mark_simulated_layout(gm: GraphModule) -> None:
    graph = gm.graph

    for ph in gm.graph.nodes:
        ph: Node

        if ph.op == FX.PLACEHOLDER:
            phmeta = MetaV.assert_non_structured(ph)

            if phmeta.role.is_distributed:
                layout = CODEGEN_PERMUTE_TRUE

                with graph.inserting_before(ph.next):
                    permuteback_ph = graph.call_function(
                        torch.permute,
                        (ph, get_permuteback_dims(phmeta.ndim)), {})
                ph.replace_all_uses_with(
                    permuteback_ph,
                    lambda user: user is not permuteback_ph)
            else:
                layout = CODEGEN_PERMUTE_FALSE
            set_codegen_io_layout(ph, layout)

    output: Node = next(iter(reversed(graph.nodes)))  # type: ignore
    outvals = cast(List[Node], output.args[0])

    out_layouts: List[CodegenPermuteLayout] = []

    for outval in outvals:
        # This `outval` cannot be more nested.
        outvalmeta = MetaV.assert_non_structured(outval)
        if outvalmeta.role.is_distributed:
            with graph.inserting_before(output):
                permute_outval = graph.call_function(
                    torch.permute,
                    (outval, get_permute_dims(outvalmeta.ndim)), {}
                )
                # NOTE
                # `outval` may be used by Nodes other than a single `output`,
                # so instead of `replace_all_uses`,
                # we change the `output.args/kwargs` only:
                #
                # `replace_input_with` will take care of structured inputs.
                output.replace_input_with(outval, permute_outval)

            out_layout = CODEGEN_PERMUTE_TRUE

        else:
            out_layout = CODEGEN_PERMUTE_FALSE

        out_layouts.append(out_layout)

    set_codegen_io_layout(output, out_layouts)

    # This is needed if we modify via `gm.graph` on a GraphModule instance.
    gm.recompile()


def simulate_codegen(
        modules: List[torch.nn.Module], graphs: List[Graph]
):
    """
    For debug only.
    This pass and the true codegen pass are mutually exclusive.

    For each involved inner GraphModule:
    -   Mark "codegen permute" layout
        (layout values are `CODEGEN_PERMUTE_TRUE` and `CODEGEN_PERMUTE_FALSE`)
        on all its `op=placeholder` Nodes and items of `op=output` Node.
        No matter that are `.role.is_distributed` or not.
    -   For such Nodes that are also `.role.is_distributed`, insert:
        -   de-permutation operations after `op=placeholder` Nodes
        -   permutation operations before `op=output` Nodes
        so that those data has a layout compatible for other code within this
        GraphModule.
    -   For all such Nodes:
        -   check the input tensors at runtime are contiguous
        -   make the resultant tensors contiguous at runtime
        so that the memory contiguousness aligns with all marked layouts.
    """

    for m, g in zip(modules, graphs):
        for node in g.nodes:
            node: Node

            if node.op == FX.CALL_MODULE:
                submod_path: str = node.target  # type: ignore
                submod = m.get_submodule(submod_path)
                if isinstance(submod, GraphModule):
                    mark_simulated_layout(submod)

    return modules, graphs
