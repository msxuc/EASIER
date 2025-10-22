# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Union
import pytest
import torch

import easier.core.module as esr
from easier.core.autodiff.autodiff import TangentFlowPropagator, GlobalTangentFlowPropCtx

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
    
