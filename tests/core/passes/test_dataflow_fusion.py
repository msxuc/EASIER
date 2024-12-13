# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn

import easier as esr
from easier.examples import Poisson


class Linsys(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        eqn = Poisson("small")
        cells = eqn.cells
        src = eqn.src
        dst = eqn.dst

        vset = esr.VertexSet(cells.shape[0])

        self.x = esr.VertexTensor(torch.ones((vset.nv,)).double(), vset)
        self.b = esr.VertexTensor(torch.ones((vset.nv,)).double(), vset)

        self.scatter = esr.Scatter(src, vset)
        self.gather_dst = esr.Gather(dst, vset)

        self.A = esr.EdgeTensor(torch.ones(src.shape[0],).double())

    def forward(self, x: esr.VertexTensor):
        x_dst = self.gather_dst(x)
        return self.scatter(x_dst * self.A)


def test_not_full_scatter():
    class Model(esr.Module):
        def __init__(self):
            super().__init__()
            vset1 = esr.VertexSet(3)
            vset2 = esr.VertexSet(10)

            idx1 = torch.LongTensor([0, 1, 2])
            idx2 = torch.LongTensor([2, 6, 8])

            data1 = torch.Tensor([1., 1., 1.])
            data2 = torch.Tensor([0.] * 10)
            self.v1 = esr.VertexTensor(data1, vset1)
            self.v2 = esr.VertexTensor(data2, vset2)

            self.sc1 = esr.Scatter(idx1, vset2)
            self.sc2 = esr.Scatter(idx2, vset2)
            self.g = esr.Gather(idx1, vset1)

            self.e = esr.EdgeTensor(data1)

            self.t = esr.Tensor(torch.Tensor([0.]))

        def forward(self):
            v0 = self.sc1(self.g(self.v1))
            v1 = self.sc1(self.e) + self.sc2(self.e)
            v2 = v1 + v0
            v3 = v1 * v0  # use v0 again
            self.v2[:] = v2 + v3
            self.t[:] = esr.mean(v3)

    m = Model()
    esr.compile([m])
    m()

    m_ = Model()
    m_()

    assert torch.equal(m.v2.sync(), m.v2)


def test_read_write_conflict():
    class TestModule(esr.Module):
        def __init__(self):
            super().__init__()
            self.linsys = Linsys()
            vset = self.linsys.x.easier_vset
            self.a = esr.VertexTensor(torch.ones((vset.nv,)).double(), vset)
            self.b = esr.VertexTensor(torch.ones((vset.nv,)).double(), vset)
            self.t = esr.Tensor(torch.Tensor([0.]).double())

        def forward(self):
            b = self.a * 0.5
            self.a[:] = self.linsys(self.linsys.x) + self.b
            self.b[:] = b + self.a

            self.t[:] = esr.norm(self.b) + esr.norm(self.a)

    m_to_jit = TestModule()
    m_jitted, = esr.compile([m_to_jit])
    m = TestModule()
    m()
    m_jitted()
    assert torch.equal(m.t, m_jitted.t)


def test_structured_input():
    class TestModule(esr.Module):
        def __init__(self):
            super().__init__()
            self.linsys = Linsys()
            vset = self.linsys.x.easier_vset
            self.a = esr.VertexTensor(torch.ones((vset.nv,)).double(), vset)
            self.b = esr.VertexTensor(torch.ones((vset.nv,)).double(), vset)
            self.t = esr.Tensor(torch.Tensor([0.]).double())

        def forward(self):
            a2 = self.a + 1
            b2 = self.b + 2
            # The definition of `a2, b2` will be moved into the
            # inner GraphModule enclosing `stack`,
            # and `a2, b2` are used later on `r2` so they will be returned by
            # the inner GraphModule.
            # The fused graphs look like:
            # ```
            # a := getattr['a']; b := getattr['b']
            # s, a2, b2 := call_module[
            #   graph{
            #       a^  := placeholder['a']; b^ := placeholder['b']
            #       a2^ := a^ + 1; ...
            #       s^  := stack([ ?A, ?B ], 1)
            #       _   := output[s^, a2^, b2^]
            #   }](a, b)
            # ```
            #
            # But if `stack` is not aware that its input is structured,
            # it will hold original `a, b` from the outer graph, i.e.
            # `?A, ?B := a, b` instead of `a^, b^` on the pseudo graph above.
            # At runtime, `a^, b^` is treated as not used any more and
            # will be released within that GraphModule and Nones are returned,
            # this causes `r+a2` to fail because of adding Nones.
            s = torch.stack([a2, b2], dim=-1)
            n = torch.norm(s, dim=1)
            r = self.linsys(n)
            r2 = r + a2 + b2
            self.t[:] = esr.norm(r2).norm()

    m_to_jit = TestModule()
    m_jitted, = esr.compile([m_to_jit])
    m = TestModule()
    m()
    m_jitted()
    assert torch.equal(m.t, m_jitted.t)


def test_nested_write_dependency():
    """
    We need to ensure write dependency in a nested esr.Module still have its
    side effect applied in the correct order.

    TODO currently statements in `inner()` are inlined into the outer graph.
    In the future, it may not be inlined anymore.
    """

    class Inner(esr.Module):
        def __init__(self):
            super().__init__()
            self.linsys = Linsys()
            self.inner_b = self.linsys.b

        def forward(self):
            self.inner_b.fill_(1.)

    class Outer(esr.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
            self.linsys = self.inner.linsys
            self.outer_b = self.linsys.b
            self.t = esr.Tensor(torch.tensor([0.], dtype=self.linsys.x.dtype))

        def forward(self):
            self.outer_b[:] = self.linsys(self.linsys.x)
            self.inner()
            self.outer_b.mul_(2.)
            self.t[0] = esr.mean(self.linsys.b).mean()

    # If the write in `self.inner()` is not properly arranged with outer Nodes,
    # it's not its filling value get averaged -- the average is 2.0 --
    # but the result from `.linsys()`.

    m_to_jit = Outer()
    m_jitted, = esr.compile([m_to_jit])
    m_jitted()
    assert 2.0 == m_jitted.t.item()
