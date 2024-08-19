# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#
# Passes-specific utilities
#

from collections import defaultdict
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, \
    Optional, Set, Tuple, Type, Union, MutableSet, Sequence, TypeVar, cast
from typing_extensions import OrderedDict, TypeVar, TypeGuard, Literal

import torch
from torch.fx.graph import Graph
from torch.fx.node import Node, Argument
from torch.fx.operator_schemas import normalize_function, ArgsKwargsPair

import easier.core.module as esr
from easier.core.runtime.dist_env import DistEnv, get_runtime_dist_env
from easier.core.utils import EasierJitException

_T = TypeVar("_T")


class EasierInterpreter(Generic[_T]):
    """
    Traverse the node list of each (esr.Module, fx.Graph) pair,
    interpret/evaluate each Node into a custom-defined value.
    This base class offers a variety of handlers for `torch.fx.Node`, specific
    to EASIER passes development.

    The traversal is:
    -   Read-only to the `Graph` node list. But modification to the individual
        `Node`s is permitted (but risky!)

    For each of these handlers:
    -   The output of the handler method will be recorded as the
        interpreted/evaluated value for that `Node`.

    -   Extra parameters that could ease the examination of that Node
        will be passed in at the beginning of the parameter list for certain
        handlers, e.g. the `module` parameter in `if_call_module()`:

        ```
        def if_call_module(self, module: torch.nn.Module)
            if isinstance(module, esr.Selector):
                pass
        ```
    """

    def __init__(
        self,
        modules: Sequence[esr.Module],
        graphs: Sequence[Graph],
    ) -> None:
        assert len(modules) == len(graphs)

        self.modules = modules
        self.graphs = graphs

        # Properties accessible during interpretation
        self.current_module: esr.Module
        self.current_graph: Graph
        self.current_node: Node

    def run(self) -> None:
        for i, (root, graph) in enumerate(zip(self.modules, self.graphs)):
            self.current_module = root
            self.current_graph = graph

            # Before traversing, we fix the nodes by copying them into a list,
            # in case the customized handler modifies the `graph.nodes` view.
            for node in list(graph.nodes):
                self.current_node = node
                self.for_each_node()

    def for_each_node(self) -> _T:
        """
        Customize the loop body of a `for node in graph.nodes:` loop.
        The base implementation by default:
        -   dispatches to a `self.if_xxx` handler.
        -   `self.current_node` is available during lifetime of this method.

        To add extra functionalities on `Node` and keep the default
        node-kind-based handling, implementor could call `super().for_node()`:
        ```
        class MyInterpreter(EasierInterpreter):
            def for_node(self):
                pre_handle_node(self.current_node)
                super().for_node()  # dispatch by node kind
                post_handle_node(self.current_node)
        ```
        """
        root = self.current_module
        node = self.current_node

        if node.op == FX.PLACEHOLDER:
            val = self.if_placeholder()

        elif node.op == FX.GET_ATTR:
            path = cast(str, node.target)
            submod_path, _sep, attr_name = path.rpartition(".")
            submod = root.get_submodule(submod_path)
            obj = getattr(submod, attr_name)

            if not isinstance(obj, torch.Tensor):
                raise EasierJitException(
                    "Currently we can only reference"
                    " torch.Tensor and subtypes"
                )

            val = self.if_get_attr(submod_path, attr_name, obj)

        elif node.op == FX.CALL_FUNCTION:
            val = self.if_call_function(node.target)

        elif node.op == FX.CALL_METHOD:
            # Currently we assume all methods are torch.Tensor methods
            method_func = getattr(torch.Tensor, cast(str, node.target))
            val = self.if_call_method(method_func)

        elif node.op == FX.CALL_MODULE:
            submod_path = cast(str, node.target)
            callee = root.get_submodule(submod_path)

            if isinstance(callee, esr.Module):
                raise EasierJitException(
                    "Currently esr.Modules should have been inlined"
                )

            val = self.if_call_module(callee)

        elif node.op == FX.OUTPUT:
            val = self.if_output()

        else:
            assert False, "unreachable"

        return val

    def if_placeholder(self) -> _T:  # type: ignore
        """
        The handler for Node `curent_node.op=='placeholder'`
        """
        pass

    def if_get_attr(self, submod_path: str, attr_name: str, attr_val
                    ) -> _T:  # type: ignore
        """
        The handler for Node `curent_node.op=='get_attr'`

        Args:
        -   submod_path (str): the dot-delimited string for the submod where
                this attribute is located under `current_module`.
                Will be empty string if the attribute is located at the root
                module.
        -   attr_name (str): the Python attribute name of this attribute.
        -   attr_val (Any): the attribute value retrieved from the paired root
                esr.Module that's specified by `curent_node.target: str`
        """
        pass

    def if_call_function(self, function) -> _T:
        """
        The handler for Node `curent_node.op=='call_function'`

        Args:
        -   function: the function callable

        By default dispatch to EASIER-specific call_operation handler.
        """
        return self.if_function_or_method(self.current_node.target)

    def if_call_method(self, method) -> _T:
        """
        The handler for Node `curent_node.op=='call_method'`

        Args:
        -   method: the callable of torch.Tensor method like
                `torch.Tensor.repeat`.

        By default dispatch to EASIER-specific call_operation handler.
        """
        return self.if_function_or_method(method)

    def if_function_or_method(self, op_callable) -> _T:  # type: ignore
        """
        The EASIER-specific handler for operation invocation.

        Args:
        -   op_callable:
                The operation callable, e.g. `operator.add, torch.add`
                or `torch.Tensor.repeat`.
        """
        pass

    def if_call_module(self, submod: torch.nn.Module) -> _T:  # type: ignore
        """
        The handler for Node `curent_node.op=='call_module'`

        Args:
        -   submod (obj): 
                the sub `torch.nn.Module` retrieved from the paired root
                esr.Module that's specified by `curent_node.target: str`
        """
        pass

    def if_output(self) -> _T:  # type: ignore
        """
        The handler for Node `curent_node.op=='output'`
        """
        pass


def normalize_selector_call_into_args(*args: _T, **kwargs: _T) -> _T:
    """
    Similar to `fx_normalize_function_variant_into_kwargs`,
    normalize the variant of invocation of `esr.Selector` to its single
    `tensor` parameter
    (the type of its argument value can be any type, not limited to `Node`),
    no matter the argument is passed in positionally or named.
    """
    def _pattern(tensor):
        # The parameter name "tensor" must match the definition of
        # `esr.Selector.forward()`
        return tensor
    return _pattern(*args, **kwargs)


def normalize_reducer_call_into_args(*args: _T, **kwargs: _T
                                     ) -> Tuple[_T, Optional[_T]]:
    def _pattern(tensor, *, out=None):
        # The parameter names "tensor, out" must match the definition of
        # `esr.Reducer.forward()`
        return tensor, out
    return _pattern(*args, **kwargs)


def vector_index_of(
    to_find: torch.Tensor, tests: torch.Tensor
) -> torch.LongTensor:
    """
    Vectorized version of index_of.

    E.g.
    to_find = [a,b,c], test = [c, W, b, V,U,X, a, Y]
    result  = [6,2,0]

    However, if `tests` is not unique,
    the index choice of duplications is not stable.
    """
    assert to_find.ndim == 1
    assert tests.ndim == 1

    sorted_tests, test_indexes = tests.sort()
    lowerbound_indexes = torch.searchsorted(sorted_tests, to_find)
    org_indexes = test_indexes[lowerbound_indexes]

    assert torch.equal(tests[org_indexes], to_find), "some not found"

    return org_indexes


# torch.fx constants


class FX:
    PLACEHOLDER = "placeholder"
    GET_ATTR = "get_attr"
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    OUTPUT = "output"


def _fx_normalization_arg_type_infer(arg) -> type:
    if isinstance(arg, Node):
        # For the category of FX-normalizable ops, FX doesn't allow a Node
        # implies a structure
        # (i.e. its metadata is `Tuple[TensorMeta, ...]` etc.).
        # So during FX-normalization, we can safely assume Nodes means Tensors.
        return torch.Tensor
    elif isinstance(arg, list):
        elem_types = list(map(_fx_normalization_arg_type_infer, arg))
        assert len(set(elem_types)) == 1, \
            "To omit type unification of list elements, we require that" \
            " list elements must be of the same type"
        return List[elem_types[0]]  # type: ignore
    elif isinstance(arg, tuple):
        elem_types = tuple(map(_fx_normalization_arg_type_infer, arg))
        return Tuple[elem_types]  # type: ignore
    else:
        return type(arg)


def fx_normalize_function_variant_into_kwargs(
    function_variant, args: tuple, kwargs: dict
) -> Dict[str, Argument]:
    """
    By default `fx.normalize_function` requires a list of types of
    the argument values of the function to normalize, this utility function
    simplifies the invocation of `fx.normalize_function` by accepting raw
    argument values.

    NOTE
    Specifying the _function variant_ to normalize is still a requirement.
    _Function variants_ are torch-exposed functions like `torch.sum`,
    instead of functions named like `torch.Tensor.sum`
    (these ".Tensor.sum" functions are yet to be invoked by expressions like
    `torch.rand(...).sum()` in a Tensor method style).
    """

    arg_types = tuple(map(_fx_normalization_arg_type_infer, args))
    kwarg_types = {k: _fx_normalization_arg_type_infer(v)
                   for k, v in kwargs.items()}

    pair: Optional[ArgsKwargsPair] = normalize_function(
        function_variant, args, kwargs,
        arg_types, kwarg_types,  # type: ignore
        normalize_to_only_use_kwargs=True)
    assert pair is not None, \
        "Failure to resolve the overload probably means" \
        f" the operator '{function_variant}' isn't a _direct export_" \
        " of a torch native function, but an extended wrapper" \
        " in torch Python layer." \
        " so it's out of FX normalization. Consider handling" \
        " possible overload resolution and argument normalization manually."

    return pair.kwargs


def tree_map(x, func):
    if isinstance(x, (list, tuple)):  # cover subtypes of list etc. too
        return type(x)(tree_map(a, func) for a in x)
    else:
        return func(x)


def isinst_checker(ty: Type[_T]) -> Callable[[Any], TypeGuard[_T]]:
    def _invoke(v) -> TypeGuard[_T]:
        return isinstance(v, ty)
    return _invoke


class OrderedSet(MutableSet[_T]):
    # For interfaces of MutableSet see
    # https://docs.python.org/3/library/collections.abc.html
    #
    # For (so-called "mixin" methods) __eq__ __sub__ etc.
    # by inheriting abc.MutableSet we get a default set-like implementation.
    #
    # NOTE `OrderedSet[T]` is not implicitly convertible to `typing.Set[T]`.

    def __init__(self, inits: Optional[Iterable[_T]] = None) -> None:
        self._odict: OrderedDict[_T, tuple] = \
            OrderedDict.fromkeys(inits, ()) \
            if inits is not None else OrderedDict()

    def __contains__(self, x: _T) -> bool:
        return x in self._odict

    def __iter__(self) -> Iterator[_T]:
        return iter(self._odict)

    def __len__(self) -> int:
        return len(self._odict)

    def add(self, value: _T) -> None:
        self._odict[value] = ()

    def discard(self, value: _T) -> None:
        if value in self._odict.keys():
            del self._odict[value]


class DisjointSet(Generic[_T]):
    #
    # Based on the pseudocode from Wikipedia
    # Source: https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    # Modifications have been made to adapt it to Python
    #

    def __init__(self, equal: Optional[Callable[[_T, _T], bool]] = None):
        self._parent: dict[_T, _T] = {}
        self._size: dict[_T, int] = {}
        self._rank: dict[_T, int] = {}
        if equal is None:
            self.equal = (lambda a, b: a == b)
        else:
            self.equal = equal

    def union(self, *objects: _T) -> None:
        for x in objects:
            if x not in self._parent:
                self._parent[x] = x
                self._size[x] = 1
                self._rank[x] = len(self._rank)
        if len(objects) >= 2:
            x = objects[0]
            for y in objects[1:]:
                self._union(x, y)

    def get_ordered_sets(self) -> List[OrderedSet[_T]]:
        """
        In the order an element is first-time pass into `union`
        """
        root_to_set: dict[_T, OrderedSet[_T]] = defaultdict(OrderedSet)
        for x in self._rank.keys():
            root = self._find(x)
            root_to_set[root].add(x)
        return list(root_to_set.values())

    def _find(self, x: _T) -> _T:
        """
        Find the root of element x, and the root is the first inserted one.
        """
        while not self.equal(self._parent[x], x):
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def _union(self, x: _T, y: _T) -> None:
        x = self._find(x)
        y = self._find(y)
        if self.equal(x, y):
            return
        if self._size[x] < self._size[y]:
            x, y = y, x
        self._parent[y] = x
        self._size[x] += self._size[y]
