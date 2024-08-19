# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Sequence, Tuple, Union
from typing_extensions import TypeAlias, TypeGuard

import torch
from torch.fx.node import Node

from easier.core.utils import EasierJitException


class _ScalarTypeCategory(Enum):
    INT = 1
    FLOAT = 2


@dataclass(eq=True, frozen=True)
class ScalarType:
    category: _ScalarTypeCategory
    precision: int

    # `@dataclass(eq=True, frozen=True)` adds a __hash__ implementation
    # such that `x == y` implies `hash(x) == hash(y)`.

    @property
    def is_integer(self) -> bool:
        return self.category == _ScalarTypeCategory.INT

    @property
    def is_floating_point(self) -> bool:
        return self.category == _ScalarTypeCategory.FLOAT


_INT = _ScalarTypeCategory.INT
BOOL = ScalarType(_INT, 1)
INT8 = ScalarType(_INT, 8)
INT16 = ScalarType(_INT, 16)
INT32 = ScalarType(_INT, 32)
INT64 = ScalarType(_INT, 64)

_FLOAT = _ScalarTypeCategory.FLOAT
FLOAT32 = ScalarType(_FLOAT, 32)
FLOAT64 = ScalarType(_FLOAT, 64)


def promote_scalar_types(*scalars: ScalarType):
    # Currently, scalar categories [INT, FLOAT] simply have a linear order,
    # and tuples are compared lexicographically. So we can `max` and get e.g.
    # `max((INT, 64), (FLOAT, 32)) == (FLOAT, 32)`.
    cat, prec = max((s.category.value, s.precision) for s in scalars)
    return ScalarType(category=_ScalarTypeCategory(cat), precision=prec)


_dtype2scalartype = {
    torch.bool: BOOL,
    torch.int8: INT8,
    torch.int16: INT16,
    torch.int32: INT32,
    torch.int64: INT64,
    torch.float32: FLOAT32,
    torch.float64: FLOAT64
}
_scalartype2dtype = {v: k for k, v in _dtype2scalartype.items()}


def convert_torch_dtype_to_scalar_type(dtype: torch.dtype):
    return _dtype2scalartype[dtype]


def convert_scalar_type_to_torch_dtype(scalar: ScalarType):
    return _scalartype2dtype[scalar]


class Role(Enum):
    PARTITION = auto()
    REPLICA = auto()

    @property
    def is_distributed(self):
        return self == Role.PARTITION


class ViewType(Enum):
    ALLOCATED = auto()
    DERIVED = auto()

    # Operators like torch.reshape returns a view or a copy depending on
    # the concrete argument values, which involves calculation of strides.
    # To avoid such calculation, currently we treat such results as
    # _undetermined_ views.
    # Until copies are clearly made, views in between are undetermined, too.
    #
    # On the other hand, this may introduced unneeded dependency.
    UNDETERMINED = auto()
    # TODO may be a set of all in-between undetermined-view Nodes


@dataclass
class View:
    """
    In fact, each tensor is a _view_ over a underlying _storage_ (e.g. a
    memory block).
    In most of the time, a PyTorch operator _allocates_ a brand-new storage
    as well as creates a unique view over it.
    (This is the trivial case. Commonly when we say "views", it's not about
    such cases)

    However, some special yet extremely common operators will reuse the
    storage and _derive_ new views only, such as:
    ```
    x = tensor[:]  # simple indexing
    x = tensor.transpose(dim0=2, dim1=1)
    x = tensor.squeeze(dim=0)
    ```

    In all cases, the storage is said to be "the source of view".

    Together with `ViewSrc`, we have the state space (defined in TensorMeta)
    about where the underlying storage comes from:
    ------------------------------------------------------------------------
    |              |         None       |  Node  |       (Node,int)        |
    |--------------|--------------------|--------|-------------------------|
    | ALLOCATED    |  trivial view on   |   X*   | an item of multi-result |
    |              | branch-new storage |        | op, always trivial view |
    |--------------|--------------------|----------------------------------|
    | DERIVED      |          X         |                                  |
    |--------------|--------------------|     reuse existing storages      |
    | UNDETEMINED  |          X         |                                  |
    -------------------------------------------------------------------------
    Marker X means the corresponding combination is invalid.

    Remarks:
    -   Only in the `ALLOCATED` cases, we deliberately do not store
        the producer Node if the Node does not stand for a multi-result op.
        (i.e. the case marked with X* is forbidden)

        Because any pass, including metadata propagation itself,
        can easily pick up the associated Node.
        Meanwhile, most MetaRule don't need to care about view info
        since only a few torch ops involves views.

    -   get_attr Node is always an `ALLOCATED+None` view.
        As FX will deduplicate names if multiple esr.Module attributes refer to
        the same esr.Tensor instance.
    """

    type: ViewType

    src: Union[
        None,               # ALLOCATED,            non-multi-result
        Node,               # DERIVED/UNDETERMINED, non-multi-result
        Tuple[Node, int]    # item of multi-result
    ]

    def __post_init__(self):
        # @dataclass protocol, run after an instance is created.
        #
        # Here we do simple checks on if the specified view info is valid.
        if self.type == ViewType.ALLOCATED:
            assert self.src is None or self.is_multi_result_item()

        elif self.type in [ViewType.DERIVED, ViewType.UNDETERMINED]:
            assert isinstance(self.src, Node) or self.is_multi_result_item()

        else:
            assert False, 'unreachable'

    def is_multi_result_item(self):
        return isinstance(self.src, tuple) and len(self.src) == 2 \
            and isinstance(self.src[0], Node) and isinstance(self.src[1], int)

    def derive_new_view(self, derived_from_node: Node) -> 'View':
        """
        Get a newly derived, concrete-ViewType view.

        NOTE for devs:
        Generally this should never be called with `current_node` etc.
        Only derive new view from one of the argument Nodes.
        """
        if self.type == ViewType.ALLOCATED:
            if self.src is None:
                return View(ViewType.DERIVED, derived_from_node)
            else:
                return View(ViewType.DERIVED, self.src)

        elif self.type in [ViewType.DERIVED, ViewType.UNDETERMINED]:
            # Only propagate undetermined views if the input is.
            return View(self.type, self.src)

        else:
            assert False, 'unreachable'

    def derive_new_undetermined_view(self, derived_from_node: Node) -> 'View':
        # All resultant cases are undetermined.
        if self.type == ViewType.ALLOCATED:
            if self.src is None:
                return View(ViewType.UNDETERMINED, derived_from_node)
            else:
                return View(ViewType.UNDETERMINED, self.src)

        elif self.type in [ViewType.DERIVED, ViewType.UNDETERMINED]:
            return View(ViewType.UNDETERMINED, self.src)

        else:
            assert False, 'unreachable'


@dataclass
class EasierTensorMeta:
    shape: Tuple[int, ...]
    dtype: ScalarType
    role: Role

    # the default case is a Allocated view.
    # NOTE the default view info object becomes a singleton.
    view_info: View = View(ViewType.ALLOCATED, None)

    @property
    def ndim(self):
        return len(self.shape)


# equivalent to "EasierNodeMeta"
StructuredTensorMeta: TypeAlias = Union[
    EasierTensorMeta,
    Sequence['StructuredTensorMeta']
]


def get_node_meta(node: Node) -> StructuredTensorMeta:
    # Store metadata using FX-experimental `Node.meta:dict`.
    # This may get extra benefit compared to store as Python attribute,
    # since data in `Node.meta` is aware and copied during FX transformation.
    meta = node.meta["easier_meta"]
    return meta


def set_node_meta(node: Node, meta: StructuredTensorMeta) -> None:
    node.meta["easier_meta"] = meta


def is_nonstructured_meta(meta: StructuredTensorMeta
                          ) -> TypeGuard[EasierTensorMeta]:
    return isinstance(meta, EasierTensorMeta)


def get_meta_from_ir_literal(
    x: Union[Node, int, float]
) -> StructuredTensorMeta:
    """
    Get metadata from FX _IR literals_, including:
    -   fx.Node (will be used as arguments to other Nodes)
    -   constant integers or floats

    Additionally a `torch.Tensor` instance will never be directly treated as
    an IR literal, because the instance is always stored in the corresponding
    `torch.nn.Module` and in FX Graph it's a `Node[op='get_attr']`.
    """
    if isinstance(x, Node):
        return get_node_meta(x)
    elif isinstance(x, bool):
        return EasierTensorMeta(shape=(), dtype=BOOL, role=Role.REPLICA)
    elif isinstance(x, int):
        # PyTorch Python wrapper isn't aware of Python int precision,
        # so we treat ints as current minimum int32 dtype
        # so they are compatible with any torch tensor with int-kind dtype.
        return EasierTensorMeta(shape=(), dtype=INT32, role=Role.REPLICA)
    elif isinstance(x, float):
        # Same as int32, treat Python float as current minimum float32.
        return EasierTensorMeta(shape=(), dtype=FLOAT32, role=Role.REPLICA)
    else:
        # NOTE for types that cannot explicitly appear on `Node.args`,
        # (`torch.Tensor` is one of such types), their metadata is always
        # propagated and carried by their corresponding `Node[op='get_attr']`.
        # We don't expect to see them here.
        raise EasierJitException(f'Value {x} cannot have associated metadata')
