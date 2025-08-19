# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from types import EllipsisType
from typing import List, Optional, Sequence, Tuple, TypeAlias, Union
import functools
import copy

import torch

from easier.core.runtime.data_loader.utils import \
    NormalizedSlice
from easier.core.runtime.dist_env import \
    get_default_dist_env
from easier.core.runtime.utils import check_collective_equality
from easier.core.utils import EasierJitException


ATTRIBUTE_PLACEHOLDER = "easier_placeholder"

Num: TypeAlias = Union[int, float, bool]


GeneralIndex: TypeAlias = Union[
    int, slice, EllipsisType, # TODO None, torch.Tensor,
]


def _wrap_function(pre_hook, post_hook, func):
    # NOTE Python captures capsule by stackframe, a dedicated function like
    # this is required, in case this is called wihtin a loop.
    @functools.wraps(func)
    def wrapper(this, *args, **kwargs):
        if pre_hook is not None:
            pre_hook(this, *args, **kwargs)
        res = func(this, *args, **kwargs)
        if post_hook is not None:
            res = post_hook(this, res, *args, **kwargs)
        return res
    return wrapper


"""
A quick template for derived DataLoader classes.
```
class DerivedDataLoader(DataLoaderBase):
    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        raise NotImplementedError()
    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    def fully_load(self, device: torch.device, replicated) -> torch.Tensor:
        raise NotImplementedError()
    def __repr__(self) -> str:
        raise NotImplementedError()
    
    # overridable:
    def collective_init(self) -> None:
        return super().collective_init()
    def minmax(self, index: NormalizedSlice) -> Tuple[Num, Num]:
        return super().minmax(index)
    def count_unique(self, index: NormalizedSlice) -> int:
        return super().count_unique(index)
```
"""


class DataLoaderBase:
    """
    The data loader for one specified data source, e.g. a HDF5 dataset.

    Calls to every method should be collective.

    NOTE for subclasses:
    Subclasses should implement each method in a way that suits the use case,
    for example, to load a range we should avoid loading-all-then-slicing.
    
    But this is not always possible, an extreme example may be:
    ```
    class RandomSamplerLoader(DataLoaderBase):
        def __init__(self, inner: DataLoaderBase):
            self.inner = inner
        def partially_load_by_range(self, region):
            random_idx = randint([for dimlen in region])
            return self.inner.partially_load_by_index(random_idx)
    ```
    where we cannot keep the simplicity of range and must fallback to load
    by index tensor.
    """

    def __init__(self) -> None:
        """
        The constructor should do simple member data storage and local tasks.

        When collective operations are needed, implementations should use
        `get_default_dist_env` because the constructors are called by users
        before `esr.compile()`.

        NOTE subclasses should not put communication, especially not call
        `coll_check_dtype_shape_devicetype` etc. in constructors.
        As DataLoaders are not only used by end users (where collective check
        makes sense) but also used by internal passes (where it's not
        collective at all).
        """
        self.shape: Tuple[int, ...]
        self.dtype: torch.dtype

        # The device on which the data loader is intially defined.
        # This device configuration only take effect with "torch" JIT backend.
        self.device: torch.device

        # e.g. "(Module).(a.b.c:Selector).idx"
        # Decided during `esr.compile()`
        self.easier_hint_name: str

    def __init_subclass__(cls) -> None:
        """
        Before a data loader API provided by a subclass runs,
        any _prefilter_ and _postfilter_ defined in this DataLoaderBase will
        run to check the environment and ensure critical requirements are
        satisfied.
        """
        for member_name, member in list(cls.__dict__.items()):
            # cls.__dict__ doesn't contain inherited methods from DistEnv
            if callable(member):
                # both `member` and `pre/post_hook` function objects are not
                # bound to some DataLoader instance yet, the `self` argument
                # will be included at the head in `args` in the wrapper.
                pre_hook = getattr(
                    DataLoaderBase, '_pre_' + member_name, None
                )
                post_hook = getattr(
                    DataLoaderBase, '_post_' + member_name, None
                )
                setattr(
                    cls, member_name,
                    _wrap_function(pre_hook, post_hook, member)
                )

    def coll_check_dtype_shape_devicetype(self):
        check_collective_equality(
            f"Tensor properties of {self.easier_hint_name}",
            [self.dtype, self.shape, self.device.type]
        )

    def collective_init(self) -> None:
        """
        Validate if the the data of this data loader is collectively correct.

        Mainly to validate DataLoaders defiend by users.
        DataLoaders created by EASIER internally generally do not need this.

        Require callers (i.e. collective_initialization pass)
        to first ensure the data loders among workers are
        actually referring to the same data set i.e. of the same type.
        """
        # Default implementation:
        self.coll_check_dtype_shape_devicetype()

        check_collective_equality(
            f"Representation of {self.easier_hint_name}",
            repr(self)
        )

    def minmax(self, index: NormalizedSlice) -> Tuple[Num, Num]:
        """
        Get minimum and maximum value within the `index` range along dim-0
        of the data source.
        
        `index` must be collectively same on all ranks.
        And must have `len(index)` > 0.

        Remarks:
        -   Note the scenarios of minmax()/count_unique(), they are 
            used by analysis and validation for Selector/Reducer.idx:

            -   Most cases are 1-d, so calculation for NormalizedSlice-on-dim-0
                cases may suffice;

            -   All contents are contained in the minmax/unique calculation,
                despite the `index` parameter, it may still OOM on rank,
                subclass implementation of these methods should do partition.
        
        -   minmax/count_unique() may recursively call minmax/count_unique(),
            but as the call stack of minmax()s serve for analysis,
            at each callstack frame the input `index` must be collectively
            same.

        -   For cases that NormalizedSlice-on-dim-0 assumption no longer holds,
            callers should load the data by themselves and call
            torch.aminmax() etc. to calcuate the result in the precise region.

            Some examples when this API is not suitable:
            -   StridedDataLoader indexes by an int on dim-0, discarding dim-0
                totally.
            -   ReshapeDataLoader receives dim-0 index (may be from outside
                StridedDataLoader) and inner dim-0&1 are flattened.
        
        TODO make all NormalizedSlice to Sequence[NormalizedSlice], then it's
        possible to partially load if some k-dim is extremely long --
        especially when Reshape-/Transpose-DataLoader are added.
        """
        # Default implementation:
        # TODO use H5DataLoader-style by-chunk calculation, otherwise on
        # a whole big data source, loading once will OOM
        # TODO only rank-0 has nonempty load_range(index != empty).
        part = self.partially_load_by_range(index)
        min_tensor, max_tensor = part.aminmax()
        amin = min_tensor.item()
        amax = max_tensor.item()
        return amin, amax
    
    def _pre_minmax(self, index: NormalizedSlice):
        assert index.count > 0, 'caller should handle empty cases separately'
        check_collective_equality('minmax index', index)
    
    def count_unique(self, index: NormalizedSlice) -> int:
        """
        Count unique elements in the exact `region` range of data source.
        
        `index` must be collectively same on all ranks.

        Used by Reducer.set_fullness()
        """
        # Default implementation:
        part = self.partially_load_by_range(index)
        _, c = part.unique(return_counts=True)
        return c

    def _pre_count_unique(self, index: NormalizedSlice):
        check_collective_equality('count unique index', index)
    

    def to(
        self,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str]] = None
    ) -> 'DataLoaderBase':
        """
        Return a cloned DataLoader with the specified properties changed.
        """
        clone = copy.deepcopy(self)
        if dtype is not None:
            clone.dtype = dtype
        if device is not None:
            clone.device = torch.device(device)
        return clone

    def __getitem__(
        self, index: Union[GeneralIndex, Tuple[GeneralIndex, ...]]
    ) -> 'DataLoaderBase':
        from easier.core.runtime.data_loader.ops import StridedDataLoader

        if not isinstance(index, tuple):
            index = (index,)

        ndim = len(self.shape)
        if len(index) > ndim:
            # TODO such cases may be valid if index None is allowed.
            raise IndexError(
                f"Too many indices for input with ndim=={ndim}"
            )

        # As long as `i, idx in enum(index)` is not out-of-range,
        # we can let StridedDataLoader.__init__ to validate each index.
        view_indices: List[Union[int, NormalizedSlice]] = []
        for i, idx in enumerate(index):
            dimlen = self.shape[i]
            if isinstance(idx, int):
                normalized_idx = idx
                if idx < 0:
                    normalized_idx = dimlen + idx
                view_indices.append(normalized_idx)

            elif isinstance(idx, slice):
                ns = NormalizedSlice.from_slice(dimlen, idx)
                view_indices.append(ns)

            elif idx is Ellipsis:
                ns = NormalizedSlice(dimlen, 0, 1, dimlen)
                view_indices.append(ns)

            else:
                # TODO support idx==None
                raise IndexError(f"Unexpect index {idx}")
        
        # TODO certain DataLoader stack can be reduced and simplified
        return StridedDataLoader(self, view_indices)


    def partially_load_by_range(self, index: NormalizedSlice) -> torch.Tensor:
        """
        Collectively load a part of the target dataset with the
        specified region.
        Each rank provides its own `index` and it's different from others'.

        Callers should make the argument `index` have in-range, positive-int
        values regarding the callee DataLoader.
            
        Args:
        -   index: NormalizedSlice
            Currently on dim-0 only

        Returns:
        - torch.Tensor: the loaded part, always on CPU
        """
        raise NotImplementedError()


    def _post_partially_load_by_range(self, res, index):
        assert res.device.type == 'cpu'
        return res

    def partially_load_by_index(self, index: torch.Tensor) -> torch.Tensor:
        """
        Collectively load a part of the target dataset with the
        specified index tensor.
        Each rank provides its own `index` and it's different from others'.
        The index is defined in the global index space.

        Args:
        - index: should always be on CPU
            Currently only 1-d and on dim-0 only

        Returns:
        - torch.Tensor: the loaded part, always on CPU
        """
        raise NotImplementedError()

    def _pre_partially_load_by_index(self, index):
        assert index.device.type == 'cpu'
        assert len(index.shape) == 1

    def _post_partially_load_by_index(self, res, index):
        assert res.device.type == 'cpu'
        return res

    def fully_load(
        self, device: torch.device, replicated: bool
    ) -> torch.Tensor:
        """
        Collectively and fully load the dataset on all ranks,
        typically for compile backend=='none' case.

        Args:
        - device: the device to load data to. (self.device will not be used.)

            NOTE
            Unlike other load methods where data is always loaded to
            CPU for JIT-internal use, after full load the data is expected
            to be ready on user-specified device, immediately.
            So instead of going via CPU, fully_load() accepts the argument
            for the fianl device.

        - replicated: if True, load the same, full data to all ranks,
            otherwise only load to rank-0, other ranks will have the
            placeholder tensor and should not do calculation with it.

            NOTE
            With placeholder tensors, other ranks can still get proper
            shapes/dtypes.

        Returns:
        - torch.Tensor: the full tensor, on the specified device.
        """
        raise NotImplementedError()

    def get_placeholder(self, device=None) -> torch.Tensor:
        """
        Allocate a new placeholder torch.Tensor of the same dtype/shape/device
        but generally consuming no memory to be compatible with cases where
        torch.Tensor object and information is needed.

        Any subclass implementation should add the attribute
        "easier_placeholder" to indicate the result is a placeholder, too.

        The placeholder is needed mainly to:
        1.  ease the inspection of tensor properties like dtype/shape/device,
            especially for the metadata pass.
        2.  fulfil `esr.Tensor.__new__(cls, data)` where the underlying data
            tensor should be set.
        3.  when backend=='none', distributed data is only loaded on rank-0,
            create placeholders on other ranks to ease retrival shape/dtype
            especially to create receiving buffers in communication.
        """
        if device is None:
            device = self.device

        # torch.Tensor.expand can expand shape-(1,) to e.g. shape-(0,0,0),
        # but not to ndim=0 shape ().
        if len(self.shape) == 0:
            ph = torch.zeros((), dtype=self.dtype, device=device)
        else:
            # TODO using `.expand()` is compatible (i.e. works with
            # `Tensor.data=ph` and propagates shape/dtype) and simple.
            #
            # However, `torch.nn.Parameter.__new__` and
            # `torch.nn._ParameterMeta(torch._C._TensorMeta)`
            # may have indicated the protocol to make a very customized object
            # to be compatible.
            #
            # That would be good because we can get totally ride of OOM,
            # not only by `.to()`, and also prevent `esr.Tensor` from e.g.
            # `torch.ones_like()`.
            ph = torch.zeros(
                (1,), dtype=self.dtype, device=device
            ).expand(self.shape)  # can even expand to (0,0,0)

        ########################################
        #              WARNING
        ########################################
        # .to() method on placeholder tensors must be strictly disabled,
        # otherwise it might materialize the memory and cause OOM!
        def _to_forbidden(self, *args, **kwargs):
            raise EasierJitException("Cannot call .to() on placeholders!")
        ph.to = _to_forbidden.__get__(ph)

        setattr(ph, ATTRIBUTE_PLACEHOLDER, True)
        return ph

    def __repr__(self) -> str:
        """
        When possible, return a string which could be treated as valid Python
        code to construct this data loader, except for `.device`, e.g.
        ```
        ArangeTensorLoader(start=0, end=1, step=1, dtype=torch.float64)
        ```

        NOTE this repr str will be used to validate compilation cache for
        mesh consistency across EASIER sessions.
        """
        raise NotImplementedError()


"""
TODO

Passes like tensor_partitioning and sparse_encoding (and potentially codegen)
leverages properties of idx tensors like being arange-d, being bounded or
being ordered to boost index calculation. The idx tensors are all originally
loaded by DataLoaders.
However, DataLoader internal methods currently return Tensors only,
therefore we didn't recognize such properties in the 1st place.
And the passes themselves maintain then discard the record objects that
indicate such properties, without making a global effort to leverage it.

Given the data-oriented nature of EASIER AOT, we may make every subsystem in
AOT return symbolic representation of data.
It's not the `pass.py` itself, but the interpreter in another layer to evaluate
the symbolic representation, deciding how to do the transformation on data.
To the extreme extent, (the symbolic representation of) simple DataLoaders
like esr.arange might get kept, and in codegen each thread can calculate data
from thread id rather than loading arange-d data from memory.
Nonetheless, CUDA acceleration, GC in AOT can also be handled by it, authors
of passes can focus on compilation logic.

That said, because of the need to balance computation costs for general cases
and the requirement of managing synchronization points of collective
communication APIs,
we may fallback to tensor calculation once when the "generalness" occur again
during AOT.

P.S. DataLoaders themselves and the AOT passes that use torch vectorized
operators (which opened a space for CUDA acceleration) are already similar
approaches, but using different symbol sets from easier.runtime.data_loader
or of torch operators.

NOTE A critical point is that when materializing data from the
symbolic representation, we do not want the symbolic engine to generate items
one-by-one and on CPU side. Possible directions for data preparation:
convert sym.rep. to torch vectorized ops, or e.g. sympy+cupy.
"""
