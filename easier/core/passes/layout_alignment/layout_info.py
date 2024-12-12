# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union, cast
from typing_extensions import TypeAlias

from torch.fx.node import Node


#
################################
# TODO  Align with codegen pass
# ==============================

def get_codegen_io_layout(node: Node) -> Optional[Union[bool, List[bool]]]:
    """
    Codegen adds layout specifications on GraphModules 'placeholder' Nodes
    and items of 'output' Nodes.

    The key is temporarily aligned with the codegen pass.
    """
    return node.meta.get('easier_codegen_layout', None)


def set_codegen_io_layout(node: Node, layout: Union[bool, List[bool]]):
    node.meta['easier_codegen_layout'] = layout

# ==============================
# TODO  Align with codegen pass
################################
#


class PermuteLayout(Enum):
    TRUE = True
    FALSE = False


StructuredLayout: TypeAlias = Union[
    PermuteLayout,
    Sequence[PermuteLayout]  # currently no more nested
]


KEY__IS_CODEGEN_NODE = 'easier_layoutAlignment_isCodegenNode'
KEY__PROPAGATED_LAYOUT = 'easier_layoutAlignment_layout'


def get_node_layout(node: Node) -> Optional[StructuredLayout]:
    """
    Get the layout specification propagated to `node: Node`.

    Returns:
    -   None: `node` has no associated layout
    -   other: the layout associated to `node`
    """
    return node.meta.get(KEY__PROPAGATED_LAYOUT, None)


def set_node_layout(node: Node, layout: Optional[StructuredLayout]):
    node.meta[KEY__PROPAGATED_LAYOUT] = layout


def mark_codegen_node(node: Node) -> None:
    node.meta[KEY__IS_CODEGEN_NODE] = True


def is_codegen_node(node: Node) -> bool:
    """
    A _codegen node_ is like the Node[op=call_module] to the kernel,
    whose output tensors are instrinsically in the specified layout.
    So no extra rewriting is needed for those Nodes.
    """
    return node.meta.get(KEY__IS_CODEGEN_NODE, False)
