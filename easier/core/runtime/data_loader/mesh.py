# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from types import EllipsisType
from typing import Iterator, List, Literal, Optional, Sequence, Tuple, TypeAlias, Union, cast
import h5py
import functools
import copy

import numpy as np
import sympy
import torch

from easier.core.runtime.data_loader.base import \
    DataLoaderBase, RegionIndex, Num
from easier.core.runtime.data_loader.factories import \
    ArangeTensorLoader, FulledTensorLoader
from easier.core.runtime.data_loader.ops import \
    CartesianProductDataLoader, StridedDataLoader, ConcatDataLoader
from easier.core.runtime.data_loader.utils import \
    get_strides

from easier.core.runtime.dist_env import \
    get_default_dist_env, get_runtime_dist_env
from easier.core.runtime.utils import check_collective_equality
from easier.core.utils import EasierJitException


# Make Mesh a nn.Module so that EASIER can look through to get DataLoaders
# that are Mesh's attributes.
class Mesh(torch.nn.Module):
    """
    Define a regular N-d mesh whose vertices are the cartesian product of
    N input arrays, the i-th input array has the shape of `(L_i,) + DIMS_i`
    (If original `DIM_i` is empty then it's treated as `(1,)`).

    The resultant shape will be like `( L_0*...*L_{N-1} ,) + DIMS`,
    and `DIMS` is a tuple and its content depends on `form`.

    Args:
    -   form ('flatten' or 'stack):
        -   'flatten': the default option.
            Subtensors for vertices in each dimensions are first
            flattened to 1-d then concat-ed.
    
            Aforementioned resultant `DIMS` will be
            `( prod(DIMS_1)+...+prod(DIMS_{N-1}), )`.

            This is the only possible organization if input arrays are not
            all homogeneous, and is the default option.

        -   'stack': only allowed if all DIMS_i are the same.
            Aforementioned resultant `DIMS` will be
            `(N,) + DIMS_0`.

    Remarks:
    -   In case of each input array carries different vertex properties,
        the subtensors for vertices are always flattened firstly
        i.e. `prod(DIMS_i)`.
    """
    def __init__(
        self,
        *dimensional_vertices: DataLoaderBase,
        form: Literal['flatten', 'stack'] = 'flatten',
        device: Union[torch.device, str, None] = None
    ):
        if device is None:
            # TODO like torch.set_default_device()
            device = 'cpu'
        self.device = torch.device(device)

        ndim = len(dimensional_vertices)
        if ndim == 0:
            raise ValueError("Must have at least one input data")

        # number of vertices per dim
        nvs: List[int] = []
        # number of hypercubes per dim
        ncs: List[int] = []
        for dt in dimensional_vertices:
            if isinstance(dt, (ArangeTensorLoader, FulledTensorLoader)):
                # TODO support general DataLoader like H5 and InMemTensor.
                raise NotImplementedError(
                    "only support easier.arange/linspace/full/ones/zeros"
                )

            if not len(dt.shape) >= 1:
                raise ValueError("Input data must be at least 1-d")
            dim_nv = dt.shape[0]
            if dim_nv == 0:
                raise ValueError("Input data must not be empty")

            nvs.append(dim_nv)
            ncs.append(dim_nv - 1)
        
        self._nvs = nvs
        self._ncs = ncs

        self.nv: int = math.prod(nvs)

        """
        Given a N-d hypercube volume at (i_1, i_2, ..., i_N) in the mesh
        whose i-d edge has L_i hypercubes (L_i + 1 vertices):
        -   it has 2N faces, each face is a (N-1)-d hypercube.
            e.g. 2d rect has 4 edges, 3d cube has 6 faces.
        -   the total number of interior faces is
            $ 2N * \prod_i {L_i} - 2 * \sum_i { \prod_{j!=i}{ L_j } } $
            or
            $ 2 * \sum_i { (L_i - 1) * \prod_{j!=i}{ L_j } }$
        """
        nfaces = math.prod(ncs) * 2 * ndim
        nbfaces = sum(math.prod(ncs[:i] + ncs[(i+1):]) for i in range(ndim))
        self.ne: int = nfaces - nbfaces

        # TODO
        # We may assume hypercubes/cells are innermost flattened,
        # to apply src/dst Selectors, we must ensure from vertices+get_index
        # user could construct cell data in the exactly innermost order.

        srcs = []
        dsts = []
        for i in range(ndim):
            interior_facets = CartesianProductDataLoader([
                ArangeTensorLoader(0, ncs[j], 1, torch.int64, device)
                if i != j else
                ArangeTensorLoader(0, ncs[i] - 1, 1, torch.int64, device)
                for j in range(ndim)
            ])
            # TODO insert Flatten to decoupling flattened indexing?
            srcs.append(
                MeshOneDimInteriorFaceIdxDataLoader(i, interior_facets, True)
            )
            dsts.append(
                MeshOneDimInteriorFaceIdxDataLoader(i, interior_facets, False)
            )
        
        # shape=(ne, ND)
        self.src = ConcatDataLoader(srcs)
        self.dst = ConcatDataLoader(dsts)
        assert self.src.shape == (self.ne,)
        assert self.dst.shape == (self.ne,)

        # flattened cartesian product of all arg dataloaders.
        # shape=(nv, ND)
        vertices = CartesianProductDataLoader(dimensional_vertices, form)
        self.vertices = vertices
    
    def get_index(self, *indices: Union[int, slice, EllipsisType]) -> DataLoaderBase:
        """
        Since EASIER requires vertices in a mesh to be organized as 1-d list,
        user can call `mesh.get_index(*IDX)` to get an index data for
        EASIER program to reconstruct indexing using coordinates in
        traditional N-d array for the mesh/vertices, i.e.:
        ```
        mesh = esr.Mesh(d0, ..., d{N-1})
        idx = mesh.get_index(idx_0, ..., idx_{N-1})
        vdata = mesh.vertices[idx]
        # equals to
        ndarray = torch.cartesian_prod(
                d0, ..., d{N-1}
            ).reshape(L0, ..., L{N-1}, -1)  # L{i} = d{i}.shape[0]
        vdata = ndarray[idx_0, ..., idx_{N-1}]
        ```

        For example, to calculate the distances between vertices along dim-2:
        ```
        def __init__(self):
            self.mesh = esr.Mesh(d0, d1, d2)

            end_vertices_idx = self.mesh.get_index(0, 0, 1:)  # for dim-2
            start_vertices_idx = self.mesh.get_index(0, 0, :-1)

            self.end_vertices_selector = Selector(end_vertices_idx)
            self.start_vertices_selector = Selector(start_vertices_idx)

        def forward(self):
            dim2_distances = \
                self.end_vertices_selector(self.mesh.vertices) \
                - self.start_vertices_selector(self.mesh.vertices)
        ```

        Remarkably, the dimensions for vertex data in all source lists
        are not supported by this method. Users have to manually index on those
        dimensions in addition to the index of vertex coordinates.

        TODO Looks a bit rigid that users must define so many idx/selector
        fields. How about allowing directly indexing batch dim using
        DataLoaders? If detected we can insert Selector for it (and can share
        Selector instances).
        TODO arguably Reducer won't be symmetrically benefited from syntactic
        sugar like this, as Reducer is more configureable and there seems no
        torch ops for Reducer as concise as getitem for Selector.
        (torch.index_reduce_ seems to exactly match Reducer)

        """
        if not (len(indices) <= len(self._nvs)):
            raise ValueError(
                "Indices must not be more than dimensions of vertices"
            )

        for idx in indices:
            if not (isinstance(idx, (int, slice)) or idx is Ellipsis):
                # TODO None -- which unsqueezes dimensions -- is not supported.
                raise TypeError("Index must be int, slice or Ellipsis")

        idx_dls: List[DataLoaderBase] = []
        for dim, nv in enumerate(self._nvs):
            idx_dl = ArangeTensorLoader(0, nv, 1, dtype=torch.int64, device=self.device)
            if dim < len(indices):
                idx = indices[dim]
                idx_dl = StridedDataLoader(idx_dl, idx)
            idx_dls.append(idx_dl)
        return CartesianProductDataLoader(idx_dls)



class MeshOneDimInteriorFaceIdxDataLoader(DataLoaderBase):
    R"""
    Given a N-d hypercube volume at (i_1, i_2, ..., i_N) in the regular mesh
    whose i-d edge has L_i hypercubes (L_i + 1 vertices):
    -   it has 2N faces, each face is a (N-1)-d hypercube.
        e.g. 2d rect has 4 edges, 3d cube has 6 faces.
    -   the total number of interior faces is
        $ 2N * \prod_i {L_i} - 2 * \sum_i { \prod_{j!=i}{ L_j } } $
        or
        $ 2 * \sum_i { (L_i - 1) * \prod_{j!=i}{ L_j } }$
    
    TODO this is basically a "mapped" DataLoader. A mapped DataLoader may
    either elementwise on all Tensor items or treat it as batch dim + k-dims.
    Then we can simply apply a chain of torch ops on load methods.
    """
    def __init__(
        self,
        # Along which dim are we traversing the faces
        dim: int,
        # A facet is where two interior faces contact
        interior_facets: CartesianProductDataLoader,
        # A reference flag for which one of the two faces
        direction: bool,
        device: Union[torch.device, str] = 'cpu'
    ):
        super().__init__()

        self.ndim = len(self.interior_facets.components)
        assert len(interior_facets.shape) == self.ndim + 1
        assert interior_facets.shape[self.ndim] == self.ndim

        self.interior_facets = interior_facets

        self.dtype = torch.int64
        self.shape = (math.prod(interior_facets.shape),)
        self.device = torch.device(device)

        self.dim = dim
        self.dimlen = interior_facets.shape[dim]  # L_i - 1

        # A reference direction along/against the dimension.
        self.direction = direction
    
    def fully_load(self, device: torch.device, replicated: bool) -> torch.Tensor:
        # (L_0*...*L_{N-1}, N) -- the last N is for coordinates.
        facets = self.interior_facets.fully_load(device, replicated).reshape(-1, self.ndim)

        # upstream/downstream cube IDs around the facet
        strides = get_strides(facets.shape)  # == (N, 1)

        up_cubes = (facets * strides).sum()

        down_cube_coords = facets.clone()
        down_cube_coords[:, self.dim] += 1
        down_cubes = (down_cube_coords * strides).sum()

        # TODO interleaving? Which kind is easier for other load methods?
        if self.direction:
            return torch.concat([up_cubes, down_cubes])
        else:
            return torch.concat([down_cubes, up_cubes])

