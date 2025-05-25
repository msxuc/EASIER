# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import dataclasses
import enum
from typing import \
    Callable, List, Sequence, Tuple, Type, TypeVar, Union, overload
from typing_extensions import TypeAlias

import torch
from torch.fx.node import Node

from easier.core.passes.utils import tree_map
from easier.core.utils import EasierJitException


KEY__METADATA_RUNTIME = 'easier_metadata_runtimeTensorMeta'
KEY__METADATA_VIEW_SRC = 'easier_metadata_viewSrc'


class Role(enum.Enum):
    # Does not have batch dim
    REPLICATED = 0

    # Has batch dim, if batch size == 0 the Node is not runnable
    # (but if it's HaloExchanger, even if batch size == 0 it must be run)
    DISTRIBUTED = 1


@dataclasses.dataclass(frozen=True, eq=True)
class RuntimeTensorMeta:
    """
    Runtime metadata is associated with a Tensor instance.

    In contrast to StaticNodeMeta, at runtime we may encounter with cases
    that a single PyTorch operator returns a tuple/list of Tensors
    (e.g. `maxval, maxpos = torch.max(dim=1)`).

    Therefore we need to be aware of the potential nested structure of
    runtime metadata.
    """
    role: Role
    shape: Tuple[int, ...]
    dtype: torch.dtype


StructuredTensorMeta: TypeAlias = Union[
    RuntimeTensorMeta,
    Sequence['StructuredTensorMeta']
]


@dataclasses.dataclass(frozen=True, eq=True)
class ViewSrc:
    """
    `ViewSrc` are Nodes that allocate the memory blocks for the tensors.

    Also, some certain multi-result operation/Node might allocate multiple
    memory blocks/tensors. Each of these memory blocks are identified with
    an extra index for the resultant item.

    For example, if one op returns/allocates two individual tensors, subsequent
    reads/writes on one tensor will not be regarded as dependent on
    reads/writes of the other tensor. Because their view_srcs are different
    in the extra index.

    To diferentiate the Tuple[Node, int] for one result of multi-result op
    from Tuple[ViewSrc] for the whole nested structure,
    make ViewSrc a distinct dataclass.
    """
    src: Union[
        Node,              # non-multi-result op, including get_attr
        Tuple[Node, int],  # item of multi-result Nodes
    ]

StructuredTensorViewSrc: TypeAlias = Union[
    ViewSrc,
    Sequence['StructuredTensorViewSrc'],

    # NOTE we are treating scalars as ()-shape tensors, those scalars do not
    # have allocators, and set `view_src = None` for them
    None
]


_T = TypeVar('_T')
_TSentinel = TypeVar('_TSentinel')


@overload
def collect_meta(
    meta: StructuredTensorViewSrc,
    f: Callable[[ViewSrc], Union[_T, _TSentinel]] = lambda x: x,
    # use other sentinel value if None is desired.
    sentinel: _TSentinel = None
) -> List[_T]: ...

@overload
def collect_meta(
    meta: StructuredTensorMeta,
    f: Callable[[RuntimeTensorMeta], Union[_T, _TSentinel]] = lambda x: x,
    # use other sentinel value if None is desired.
    sentinel: _TSentinel = None
) -> List[_T]: ...

@overload
def collect_meta(
    meta: object,
    f: Callable[[_T], Union[_T, _TSentinel]] = lambda x: x,
    # use other sentinel value if None is desired.
    sentinel: _TSentinel = None,
    *,
    leaf_type: Type[_T]
) -> List[_T]: ...

def collect_meta(meta, f = lambda x: x, sentinel = None, leaf_type = (RuntimeTensorMeta, ViewSrc)) -> list:
    ys = []

    def _collect(x):
        if isinstance(x, leaf_type):
            y = f(x)
            if y is not sentinel:
                ys.append(y)

    _ = tree_map(meta, _collect)

    return ys


def set_node_meta(node: Node, tensor_meta: StructuredTensorMeta):
    node.meta[KEY__METADATA_RUNTIME] = tensor_meta


def get_node_meta(node: Node) -> StructuredTensorMeta:
    return node.meta[KEY__METADATA_RUNTIME]


def get_runtime_metadata_from_scalar(
    val: Union[bool, int, float]
) -> RuntimeTensorMeta:
    if isinstance(val, bool):
        return RuntimeTensorMeta(Role.REPLICATED, (), torch.bool)
    elif isinstance(val, int):
        # PyTorch Python wrapper isn't aware of Python int precision,
        # so we treat ints as current minimum int32 dtype
        # so they are compatible with any torch tensor with int-kind dtype.
        return RuntimeTensorMeta(Role.REPLICATED, (), torch.int32)
    elif isinstance(val, float):
        # Same as int32, treat Python float as current minimum float32.
        return RuntimeTensorMeta(Role.REPLICATED, (), torch.float32)
    else:
        # NOTE for types that cannot explicitly appear on `Node.args`,
        # (`torch.Tensor` is one of such types), their metadata is always
        # propagated and carried by their corresponding `Node[op='get_attr']`.
        # We don't expect to see them here.
        raise EasierJitException(
            f'Scalar value {val} of type {type(val)}'
            ' cannot have associated metadata'
        )


def set_node_view_src(node: Node, view_src: StructuredTensorViewSrc):
    node.meta[KEY__METADATA_VIEW_SRC] = view_src

def get_node_view_src(node: Node) -> StructuredTensorViewSrc:
    return node.meta[KEY__METADATA_VIEW_SRC]
