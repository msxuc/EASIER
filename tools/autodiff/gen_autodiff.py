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


"""
Empirically, FX-traceable PyTorch operators have following properties,
therefore can be categorized:

|------------------|-----------------------------|--------------------------|
| Operator system  |    Overloading mechanism    | FX-traceability priority |
|------------------|-----------------------------|--------------------------|
|       aten       |    CPP-like overloading     |          Low             |
|------------------|-----------------------------|--------------------------|
| torch.functional |      Python functions,      |                          |
|    functions     |       no overloading        |          High            |
|------------------|-----------------------------|--------------------------|

-   `aten` operators are PyTorch CPP/native-level operators, with explicit
    and CPP-like (by parameter types and number) overloadings, e.g.

    ```
    add.Tensor(Tensor self, Tensor other, *, Scalar alpha) -> Tensor
    add.out(Tensor self, Tensor other, *, Scalar alpha, Tensor out) -> Tensor
    add.Scalar(Tensor self, Scalar other, Scalar alpha) -> Tensor
        ~~~~~~
        may have "overloading hints"

    all(Tensor self) -> Tensor
    all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
        ~~~~~~~
        "overloading hints" do not necessarily follow a convention.
    ```

    Generally defined in `$PYTORCH/aten/src/ATen/native/native_functions.yaml`.

-   `aten` operators are also exposed by Python wrappers as Python functions or
    torch.Tensor methods, and PyTorch implements a customized overloading
    resolution mechanism within those Python callables.

    For example, when we pass the wrong arguments to an operator, PyTorch would
    raise an exception telling how it's overloaded
    (without overloading hints):

    ```
    >>> torch.all(1)
    TypeError: all() received an invalid combination of arguments - got (int),
    but expected one of:
    * (Tensor input, *, Tensor out = None)
    * (Tensor input, tuple of ints dim = None, bool keepdim = False, *,
       Tensor out = None)
    * (Tensor input, int dim, bool keepdim = False, *, Tensor out = None)
    ```

-   `torch.functional` functions are purely Python-world definitions,
    therefore these operators have no natural overloading mechanism, e.g.

    ```python
    # $PYTORCH/torch/functional.py
    def split(
        tensor: Tensor,
        split_size_or_sections: Union[int, List[int]],
        dim: int = 0,
    ) -> Tuple[Tensor, ...]:
        return tensor.split(split_size_or_sections, dim)
        # Finally it's still dispatched to `aten`-system, but not tangible to
        # torch FX.
    ```

    The key outcomes are:
    -   `torch.functional.split` etc. will get traced by FX and shadow
        `aten`-system (i.e. Python wrappers) operators like `aten::split`.
    -   The function objects `torch.functional.split` etc. cannot be recognized
        by any (PyTorch-internal) operator analysis systems like
        `torch._C._jit_get_schemas_for_operator`.

"""


@dataclasses.dataclass
class DerivativeDefinition:
    name: str
    result: Union[str, Dict[int, str]]

@dataclasses.dataclass
class ExceptionalDefinition:
    name: str
    reason: Literal[
        'no_result_rule',
        
        'torch_functional_only',
        'tensor_method_only', # TODO REALLY?
        'native_function_only',
    ]


"""
TODO
1.  tensordot is only defined in native_functions.yaml, but not mentioned in derivatives.yaml
    The behavior of functorch.jvp seems to inline aten::tensordot which is a composition of operators
    (those in native_functions.yaml and with `tags: core` -- those will remain after aten-to-aten decomposition)
"""


def _aten_name_with_overloading_suffix(fs: torch._C.FunctionSchema) -> str:
    # fs.name is without overloading suffix
    if fs.overload_name:
        name_w_ovld_hint = fs.name + '.' + fs.overload_name
    else:
        name_w_ovld_hint = fs.name
    return name_w_ovld_hint


# Get all non-internal operator signatures e.g.
# "all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"
def parse_native_functions_yaml(args: 'CliArgs') -> List[str]:
    nf_yaml_fp = os.path.join(
        args.pytorch_codebase, 'aten/src/ATen/native/native_functions.yaml'
    )

    with open(nf_yaml_fp, 'r') as yaml_fs:
        funcdefs: list = yaml.safe_load(yaml_fs)

    funcsigs = []

    for funcdef in funcdefs:
        funcdef: dict

        funcsig: str = funcdef['func']

        # torch internal operators, skip, e.g.
        # "_unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"
        if funcsig.startswith('_'):
            continue

        # Exclude Tensor-creators like zeros/eye etc
        #
        # Extreme cases are like:
        # split.Tensor(Tensor(a -> *) self, SymInt split_size, \
        # int dim=0) -> Tensor(a)[]
        # So we need to filter out Tensor in suffix and Tensor in result.
        func_type_str = '('.join(funcsig.split('(')[1:])
        param_list_str = ' -> '.join(func_type_str.split(' -> ')[:-1])
        if 'Tensor' not in param_list_str:
            continue

        # Exclude inplace operators `op_`
        name_wo_suffix = funcsig.split('(')[0].split('.')[0]
        if name_wo_suffix.endswith('_'):
            continue

        # Exclude inplace operators `op(..., Tensor(x!)) -> Tensor(x!)`
        # which would result in Python `op(out=...)` parameter.
        if '!' in param_list_str:
            continue

        funcsigs.append(funcsig)
    
    return funcsigs

def parse_derivatives_yaml(args: 'CliArgs') -> Tuple[List[DerivativeDefinition], List[ExceptionalDefinition]]:
    derivatives_yaml_fp = os.path.join(
        args.pytorch_codebase, 'tools/autograd/derivatives.yaml'
    )

    # TODO some op result item has alias like Q K V, and does not have 'result' field
    with open(derivatives_yaml_fp, 'r') as yaml_fs:
        torch_derivatives: list = yaml.safe_load(yaml_fs)

    mm = []

    jitall_names_w_sfx = []
    alls = torch._C._jit_get_all_schemas()
    for s in alls:
        if 'aten::' in s.name:
            wo_aten_ns = s.name[6:]
            if wo_aten_ns.startswith('_'):
                continue

            for arg in s.arguments:
                if 'Tensor' in arg.type.annotation_str:
                    jitall_names_w_sfx.append(_aten_name_with_overloading_suffix(s))
                    break
        
        if 'aten::mm' in s.name:
            mm.append(s)
    
    nf_sigs = parse_native_functions_yaml(args)
    nf_names_w_sfx = set('aten::' + sig.split('(')[0] for sig in nf_sigs)

    missing = nf_names_w_sfx - set(jitall_names_w_sfx)

    schemas_from_derivatives = []
    for torch_deriv in torch_derivatives:
        torch_deriv: dict

        # e.g. "all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"
        name: str = torch_deriv['name']

        # e.g. "all.dim"
        name_with_suffix = name.split('(')[0]

        if name_with_suffix.startswith('_'):
            # torch internal operators, skip, e.g.
            # "_unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"
            continue

        # Torch JIT system contains more operators like:
        # - control flow operators
        # - TorchScript VM operators
        # - Type-system-wise generic operators e.g. `add.t(t a, t b) -> t`
        # - many more ...
        # We have to filter to get 
        #
        # This _C API doesn't allow overloading suffix
        schemas: List[torch._C.FunctionSchema] = \
            torch._C._jit_get_schemas_for_operator(name_with_suffix.split('.')[0])
        
        aten_name_w_suffix = 'aten::' + name_with_suffix
        for fs in schemas:
            fs_name_w_suffix = _aten_name_with_overloading_suffix(fs)

            if fs_name_w_suffix == aten_name_w_suffix:
                break
        else:
            assert False, f"{aten_name_w_suffix} not found in TorchScript"
        
        schemas_from_derivatives.append(fs)
    



    probably_need_manually = set(jitall_names_w_sfx) - set(schemas)
        




    
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
    parser.add_argument('--pytorch_codebase', type=str)
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

    