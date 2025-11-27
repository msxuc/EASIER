# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Union
import pytest
import torch

import easier as esr
from easier.core.autodiff.autodiff import Jvp, JvpTransformer
from easier.numeric import linsys


@pytest.mark.usefixtures('dummy_dist_env')
class TestJvpTransformation:
    def test_mul_t_t(self):
        class M(esr.Module):
            def __init__(self):
                super().__init__()

                # have tangent
                self.v1 = esr.Tensor(torch.rand(10, 3).double(), mode='partition')
                self.v2 = esr.Tensor(torch.rand(10, 3).double(), mode='partition')
                self.r1 = esr.Tensor(torch.rand(3).double(), mode='replicate')

                # no tangent
                self.v3 = esr.Tensor(torch.rand(10, 3).double(), mode='partition')
                self.r2 = esr.Tensor(torch.rand(3).double(), mode='replicate')

                self.res = esr.Tensor(esr.zeros([10, 3], dtype=torch.float64), mode='partition')

            
            def forward(self):
                v2 = self.v1 * self.v2
                v3 = self.v1 * self.v3
                r1 = self.v1 * self.r1
                r2 = self.v1 * self.r2
                c1 = self.v1 * 5

                self.res[:] = v2 + v3 + r1 + r2 + c1
        
        raw = M()
        jvp_transfomer = JvpTransformer(raw, raw).run()
    
    def test_nest_arg_tangent_appear_and_not_appear(self):
        class M(esr.Module):
            def forward(self):
                return torch.concat([x1, x2, x3], dim=1)
    

@pytest.mark.usefixtures('dummy_dist_env')
class TestJvp:
    def test_spmv(self):
        nx = 30
        ny = 20

        _ne = nx * ny // 2
        nnz = torch.randint(0, nx * ny, [_ne]).unique(sorted=True)
        ne = nnz.shape[0]

        Ae = torch.rand_like(nnz, dtype=torch.float64)
        x = torch.rand(nx, dtype=torch.float64)
        y = torch.rand(ny, dtype=torch.float64)

        class SpMV(esr.Module):
            def __init__(self):
                super().__init__()

                p = torch.randperm(ne)
                nnz2 = nnz[p]
                s_idx = nnz2 % ny
                r_idx = nnz2 // nx

                self.Ae = esr.Tensor(Ae[p], mode='partition')
                self.selector = esr.Selector(s_idx)
                self.reducer = esr.Reducer(r_idx, ny)

                self.x = esr.Tensor(x, mode='partition')
                self.y = esr.Tensor(y, mode='partition')

            def forward(self):
                y = self.reducer(
                    self.selector(self.x) * self.Ae
                )
                # self.y[:] = y
        
        raw = SpMV()

        raw_jvp = Jvp()
        t_x = esr.Tensor(x, mode='partition')

        jvp_transfomer = JvpTransformer(raw, raw_jvp, { raw.x: t_x}).run()
        jvp_g = jvp_transfomer.jvp_graph


        [jvp], [tx], [ty] = esr.jvp([raw], [raw.x], [raw.y])

        # [jvp] = esr.compile([jvp], backend='torch')


        raw_f = raw.forward
        [raw] = esr.compile([raw], backend='none')

        # NoneBackendEngine adds extra computation
        raw.forward = raw_f

        def _classic(input_x: torch.Tensor):
            1




@pytest.mark.skip
@pytest.mark.usefixtures('dummy_dist_env')
class TestTangentFlowProp:
    def test_forward_prop(self):
        irrelevant = esr.Tensor(esr.arange(100, dtype=torch.float64), mode='partition')
        input = esr.Tensor(esr.arange(100, dtype=torch.float64), mode='partition')
        late_carrier_outer1 = esr.Tensor(esr.arange(100, dtype=torch.float64), mode='partition')
        late_carrier_outer2 = esr.Tensor(esr.arange(100, dtype=torch.float64), mode='partition')
        late_carrier_midonce1 = esr.Tensor(esr.arange(100, dtype=torch.float64), mode='partition')
        late_carrier_midmulti1 = esr.Tensor(esr.arange(100, dtype=torch.float64), mode='partition')

        class Inner(esr.Module):
            def __init__(self):
                super().__init__()
            1

        class Middle(esr.Module):
            def __init__(self, late_carrier_mid_1: esr.Tensor, late_carrier_mid_2: esr.Tensor):
                super().__init__()

                # InnerOnce gets called multi times in MidMulti,
                # so it won't be treated as really Module that's called once.
                self.inner_once = Inner()

                self.inner_multi =  Inner()

                self.irrelevant = irrelevant
                self.input = input
                self.late_carrier_mid_1 = late_carrier_mid_1
                self.late_carrier_mid_2 = late_carrier_mid_2
            
            def forward(self):
                _middle_outer_template(self, self.late_carrier_mid_1, self.late_carrier_mid_2,
                                       self.inner_once, self.inner_multi)

        class Outer(esr.Module):
            def __init__(self):
                super().__init__()
                self.mid_once = Middle()
                self.mid_multi = Middle()

                self.irrelevant = irrelevant
                self.input = input
                self.late_carrier_outer1 = late_carrier_outer1
                self.late_carrier_outer2 = late_carrier_outer2
            
            def forward(self):
                _middle_outer_template(self, self.late_carrier_outer1, self.late_carrier_outer2,
                                       self.mid_once, self.mid_multi)


        def _middle_outer_template(
            self: Union['Middle', 'Outer'],
            late_carrier_1: esr.Tensor, late_carrier_2: esr.Tensor,
            nested_once: esr.Module, nested_multi: esr.Module,
        ):
            v1 = torch.sin(self.input)
            late_carrier_1.add_(self.irrelevant)
            v2 = torch.cos(v1)
            late_carrier_1.add_(v2)

            nested_once()

            nested_multi()
            nested_multi()
            nested_multi()

            late_carrier_2.add_(self.irrelevant)
            v3 = torch.neg(self.input)
            late_carrier_2.add_(v3)



    def test_late_carrier_and_alias(self):
        input = esr.Tensor(esr.arange(100, dtype=torch.float64), mode='partition')
        late_carrier_outer = esr.Tensor(esr.arange(100, dtype=torch.float64), mode='partition')
        late_carrier_inner = esr.Tensor(esr.arange(100, dtype=torch.float64), mode='partition')

        class Inner(esr.Module):
            def __init__(self):
                super().__init__()

                self.input = input
                self.late_carrier_outer = late_carrier_outer
                self.late_carrier_inner = late_carrier_inner
            
            def forward(self):
                inner_v1 = self.late_carrier_outer * 3
                self.late_carrier_inner.sub_(inner_v1)


        class Outer(esr.Module):
            def __init__(self):
                super().__init__()

                self.input = input
                self.late_carrier_outer = late_carrier_outer

                self.inner = Inner()
            
            def forward(self):
                v1 = torch.sin(self.late_carrier_outer)
                self.late_carrier_outer.add_(self.input)

                self.inner()

                v2 = torch.exp(self.late_carrier_outer)
                v3 = self.inner.late_carrier_inner / 5
    
