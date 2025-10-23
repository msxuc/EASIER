# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import dataclasses
import itertools
import os
import re
from typing import Dict, List, Literal, Tuple, Union
import yaml

import torch
import torch.jit.supported_ops
import torch.fx.operator_schemas as OS

@dataclasses.dataclass
class DerivativeDefinition:
    name: str
    result: Union[str, Dict[int, str]]

@dataclasses.dataclass
class ExceptionalDefinition:
    name: str
    reason: Literal[
        'no_result_rule',
        
        # Some Python-function operators defined in torch.functional have the
        # same names as PyTorch native ATen operators, e.g. `torch.split`,
        # which would shadows the Python wrappers for those native operators,
        # especially during FX tracing, therefore needs manual handling in AD.
        #
        # P.S. The Python types might show their difference:
        # ```
        # type(torch.split) => <function split ..
        #                       ~~~~~~~~
        # type(torch.split_with_sizes) => <built-in method split_with_sizes ...
        #                                  ~~~~~~~~~~~~~~~
        # ```
        'torch_functional',
        
    ]

def parse_derivatives_yaml(args: 'CliArgs') -> Tuple[List[DerivativeDefinition], List[ExceptionalDefinition]]:
    derivatives_yaml_fp = os.path.join(
        args.pytorch_codebase, 'tools/autograd/derivatives.yaml'
    )

    # TODO some op result item has alias like Q K V, and does not have 'result' field
    with open(derivatives_yaml_fp, 'r') as yaml_fs:
        torch_derivatives: list = yaml.safe_load(yaml_fs)
    
    for torch_deriv in torch_derivatives:
        torch_deriv: dict

        # e.g. "all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"
        name: str = torch_deriv['name']

        # e.g. "all.dim"
        name_with_overloading_hint = name.split('(')[0]

        # e.g. "all"
        name_without_ovld_hint = name_with_overloading_hint.split('.')[0]

        if name_without_ovld_hint.startswith('_'):
            # torch internal operators, skip, e.g.
            # "_unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"
            continue

        if name_with_overloading_hint in torch.functional.__all__:
            # Some Python-function operators defined in torch.functional have
            # the same names as PyTorch native ATen operators,
            # e.g. `torch.split`, which would *shadows* the Python wrappers for
            # those native operators during FX tracing, and all overloadings.
            # Therefore they needs manual handling in AD.
            #
            # P.S. The Python types might show their difference:
            # ```
            # type(torch.split) => <function split ...
            #                       ~~~~~~~~
            # type(torch.split_with_sizes) => <built-in method split_with_sizes
            #                                  ~~~~~~~~~~~~~~~
            # ```
            #
            # `torch.functional` module also include FX-traceable operators
            # that do not have native counterparts at all, e.g. `torch.einsum`.
            continue
        
        # Torch JIT system contains more operators like control flow operators
        # in TorchScript, so we need to filter by 'aten' CPP namespace
        aten_name_wo_ovld_hint = 'aten::' + name_without_ovld_hint

        schemas: List[torch._C.FunctionSchema] = \
            torch._C._jit_get_schemas_for_operator(aten_name_wo_ovld_hint)
        
        aten_name_w_ovld_hint = 'aten::' + name_with_overloading_hint
        for fs in schemas:
            # fs.name is without overloading hint
            if fs.overload_name:
                fs_name_w_ovld_hint = fs.name + '.' + fs.overload_name
            else:
                fs_name_w_ovld_hint = fs.name

            if fs_name_w_ovld_hint == aten_name_w_ovld_hint:
                break
        else:
            assert False, f"{aten_name_w_ovld_hint} not found in TorchScript"
        




    
    return ([], [])
    





def parse_pushforward_info():
    """
    Given all @pushforward definitions, 

    NOTE the `x_t is None` check to omit tangent component term for
    zero input tangent will never be triggered, as during FX tracing here all
    arguments are `fx.Proxy(op='placeholder')` and not None.
    Therefore we can get all parameter names and then by checking `_t` suffix
    we know which parameters can be differentiated.
    """
    from easier.core.autodiff.pushforward import pushforward_registry


@dataclasses.dataclass
class CliArgs:
    pytorch_codebase: str
    mode: Literal['collect_op_names', 'generate_rules']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--pytorch_codebase', type=str, default='/mnt/c/Users/caox/Documents/repos/GitHub/pytorch')
    parser.add_argument(
        '--mode',
        choices=['collect_op_names', 'generate_rules'],
        default='generate_rules',
        help="""Which phase of the generation to run:

collect_op_names: Collect all `name` field of `dertivatives.yaml` into
    `names.yaml`.
    Devs can edit the name list to choose which operators are for
    `generate_rules` phase to handle.

generate_rules: Generate EASIER autodiff rules for operators in `names.yaml`.
"""
    )

    args = CliArgs(**vars(parser.parse_args()))

    # if args.mode == 'collect_op_names':
    parse_derivatives_yaml(args)

    