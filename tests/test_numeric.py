# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
import os
import pytest
import sys
import torch

import easier as esr
from easier.numeric import solver

from tests.utils import \
    have_cuda, when_ngpus_ge_2, torchrun_singlenode, \
    import_poisson, MESH, POISSON


def worker__test_cg(local_rank: int, world_size: int, backend: str):
    Poisson = import_poisson()

    # for prec_type in ['symmetric', None]:
    for prec_type in [None]:

        eqn = Poisson(MESH, POISSON)
        sol = solver.CG(eqn.A, eqn.b, eqn.x)
        [sol] = esr.compile([sol], backend=backend)  # type: ignore
        sol: solver.CG

        info = sol.solve(atol=1e-4, maxiter=1000)
        assert info["residual"] < 1e-4


def worker__test_gmres(local_rank: int, world_size: int, backend: str):
    Poisson = import_poisson()

    # for prec_type in ['forward', 'backward', 'symmetric', None]:
    for prec_type in [None]:

        eqn = Poisson(MESH, POISSON)
        sol = solver.GMRES(eqn.A, eqn.b, eqn.x)
        [sol] = esr.compile([sol], backend=backend)  # type: ignore
        sol: solver.GMRES

        info = sol.solve(atol=1e-4, maxiter=1000)
        assert info["residual"] < 1e-4


@pytest.mark.parametrize('nprocs, backend', [
    (1, 'torch'),
    (1, 'cpu'),
    pytest.param(1, 'cuda', marks=have_cuda),
    (2, 'torch'),
    (2, 'cpu'),
    pytest.param(2, 'cuda', marks=when_ngpus_ge_2),
])
class TestLinearSolver:

    def test_cg(self, nprocs: int, backend: str):
        if backend != 'cuda':
            init_type = 'cpu'
        else:
            init_type = 'cuda'
        torchrun_singlenode(
            nprocs, worker__test_cg, (backend,), init_type=init_type
        )

    def test_gmres(self, nprocs: int, backend: str):
        if backend != 'cuda':
            init_type = 'cpu'
        else:
            init_type = 'cuda'
        torchrun_singlenode(
            nprocs, worker__test_gmres, (backend,), init_type=init_type
        )
