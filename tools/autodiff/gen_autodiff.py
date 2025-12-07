# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import dataclasses
import enum
import functools
import itertools
import os
import re
from typing import Any, Dict, Generator, List, Literal, Optional, Self, Sequence, Set, Tuple, TypeAlias, TypeVar, TypedDict, Union, cast
import more_itertools
import yaml
import pyparsing

import torch

from easier.core.autodiff.autodiff_rule import Differentiability, RequiredParam
from easier.core.autodiff.utils import FxConst


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


# May include TorchScript JIT system specific ops.
schemas: Dict[str, List[torch._C.FunctionSchema]] = {}


for schema in torch._C._jit_get_all_schemas():
    if schema.name.startswith('aten::'):
        name = schema.name[6:]
        schemas.setdefault(name, []).append(schema)


def _parse_differentiability(
    schema: torch._C.FunctionSchema, cpp_expr: str, is_auto_element_wise: bool
) -> Differentiability:
    vars: List[str] = re.findall(R'([A-Za-z][A-Za-z0-9_:]*)', cpp_expr)
    assert len(vars) > 0

    diffable_params: List[str] = []
    other_params: List[Tuple[str, Union[RequiredParam, FxConst]]] = []

    # if auto_element_wise, `grad` becomes `self_t` or `original_self_t`
    for i_p, arg in enumerate(schema.arguments):
        if is_auto_element_wise and i_p == 0:
            assert arg.name == 'self', 'a convention'
            assert 'grad' in vars
            diffable_params.append('input')

        else:
            diffable = False
            for possible_tangent_var in [f'{arg.name}_t', f'original_{arg.name}_t']:
                if possible_tangent_var in vars:
                    diffable = True
        
            if arg.name == 'self':
                param_name = 'input'
            else:
                param_name = arg.name
            
            if diffable:
                diffable_params.append(param_name)
            else:
                if not arg.has_default_value():
                    default = RequiredParam()
                else:
                    default = arg.default_value
                
                other_params.append((param_name, default))  # type: ignore
    
    return Differentiability(diffable_params, other_params)


@dataclasses.dataclass
class DerivEntry:
    schema: torch._C.FunctionSchema

    # To dump to Python code directly
    differentiability: Differentiability


def parse_derivatives_yaml(
    args: 'CliArgs', # opdefs: List[OpDef], removed_incremental_inplace_ops: Set[OpDef]
    op_names: List[str]
) -> List[DerivEntry]:
    print("""
##############################
#   Parse derivatives.yaml   #
##############################
""")
    result_entries: List[DerivEntry] = []

    derivatives_yaml_fp = os.path.join(
        args.pytorch_codebase, 'tools/autograd/derivatives.yaml'
    )

    # TODO some op result item has alias like Q K V,
    # and does not have 'result' field
    with open(derivatives_yaml_fp, 'r') as yaml_fs:
        torch_derivatives: list = yaml.safe_load(yaml_fs)
    

    schemas_and_yamldefs_by_opname: \
        Dict[str, List[Tuple[torch._C.FunctionSchema, dict]]] = {}

    for yaml_derivdef in torch_derivatives:
        yaml_derivdef: dict

        # e.g.
        # clamp.Tensor(Tensor self, ...) -> Tensor
        deriv_op_sig = cast(str, yaml_derivdef['name'])

        # torch internal operators
        if deriv_op_sig.startswith('_'):
            continue

        deriv_op_name, derive_op_overload_name, *_ = \
            deriv_op_sig.split('(')[0].split('.') + ['']

        if deriv_op_name not in op_names:
            continue

        for overloading_schema in schemas[deriv_op_name]:
            # Not accepted by torch.jvp()
            if any(arg.is_out for arg in schema.arguments):
                continue

            if overloading_schema.overload_name == derive_op_overload_name:
                schemas_and_yamldefs_by_opname.setdefault(
                    deriv_op_name, []
                ).append(
                    (overloading_schema, yaml_derivdef)
                )
                break
        else:
            assert False, f'Schema not found for {deriv_op_sig}'


    print("""
#   Ops not in derivatives.yaml (custom DiffRule or Differentiability needed):
""")
    for op_name in op_names:
        if op_name not in schemas_and_yamldefs_by_opname:
            print(op_name)
        
    print("""
#   Ops in derivatives.yaml but needing manual handling
""")
    
    for _, schemas_and_yamldefs in schemas_and_yamldefs_by_opname.items():
        # For those overloading whose param names (w/ types) are the same,
        # merge Scalar-type-param version into Tensor-type-param version.
        by_param_names: Dict[
            str,  # e.g. input@other@alpha -- whatever hashable identifer
            List[Tuple[torch._C.FunctionSchema, dict]]
        ] = {}

        for kv in schemas_and_yamldefs:
            overloading_schema, yamldef = kv
            param_name_comb = '@'.join(arg.name for arg in overloading_schema.arguments)
            by_param_names.setdefault(param_name_comb, []).append(kv)
        
        def _schema_lt(
            s1: torch._C.FunctionSchema, s2: torch._C.FunctionSchema
        ):
            all_lt = []
            for a1, a2 in more_itertools.zip_equal(s1.arguments, s2.arguments):
                t1 = str(a1.type)
                t2 = str(a2.type)
                if t1 != t2:
                    if t1 == 'number' and t2 == 'Tensor':
                        all_lt.append(True)
                    elif t1 == 'Tensor' and t2 == 'number':
                        all_lt.append(False)
                    else:
                        assert False, \
                            'Not mergeable arguments are not equal in' \
                            f' {s1} and {s2}'

            assert len(all_lt) > 0, f'{s1} and {s2}'
            assert len(set(all_lt)) == 1, f'{s1} and {s2}'

            return all_lt[0]

        
        # For each group with same param names (ignoring types),
        # try to merge schema
        # param type Scalar into Tensor.
        for _, mergeable_schemas_and_yamldefs in by_param_names.items():
            upperbound = mergeable_schemas_and_yamldefs[0]

            if len(mergeable_schemas_and_yamldefs) > 1:
                # effectively form a lattice
                for candidate in mergeable_schemas_and_yamldefs[1:]:
                    if _schema_lt(upperbound[0], candidate[0]):
                        upperbound = candidate
                
            upper_schema, yaml_derivdef = upperbound
            
            if 'dispatch' in yaml_derivdef:
                yaml_derivdef = yaml_derivdef['dispatch']['Default']

            if 'result' not in yaml_derivdef:
                # Some simple and common cases are of this kind: the op itself
                # is not diff-able.
                # We need to parse the derivative.yaml entry and explicitly add
                # a non-diff-able rule for EASIER.

                # Otherwise, it is multi-res or has no tangent rule (composed),
                # we'd better skip it and add diff rule for it manually in EASIER.

                print(f'{upper_schema} does not have "result" field')
                continue

            else:
                tangent_expr: str = yaml_derivdef['result']
                tangent_expr = tangent_expr.strip()
                if tangent_expr == 'auto_linear':
                    assert upper_schema.arguments[0].name == 'self'
                    differentiability = _parse_differentiability(upper_schema, 'self_t', False)
                    result_entries.append(DerivEntry(
                        upper_schema, differentiability
                    ))
                
                else:
                        
                    # 'grad' becomes 'self_t' or 'original_self_t'
                    is_auto_element_wise = tangent_expr == 'auto_element_wise'

                    if is_auto_element_wise:
                        assert 'self' in yaml_derivdef, \
                            f'{deriv_op_sig}\n\tis auto but does not have "self"' \
                            f' tangent defined' \
                            f'\n\t{yaml_derivdef}\n' \
                        
                        tangent_expr = yaml_derivdef['self']

                    tangent_expr = tangent_expr.strip()
                    differentiability = _parse_differentiability(upper_schema, tangent_expr, is_auto_element_wise)

                    result_entries.append(DerivEntry(
                        upper_schema, differentiability
                    ))

    return result_entries


def dump_torch_jvp_differentiabilities_file(all_entries: List[DerivEntry]):
    by_names: Dict[str,  List[DerivEntry]] = {}
    for entry in all_entries:
        entries = by_names.setdefault(entry.schema.name[6:], [])
        entries.append(entry)
    
    fp = os.path.join(
        os.path.dirname(__file__),
        '../../easier/core/autodiff/torch_jvp_differentiabilities.py'
    )
    with open(fp, 'w') as fs:
        fs.write(
            """# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#
# GENERATED by tools/autodiff/gen_autodiff.py
#

import torch

from .autodiff_rule import Differentiability, required, differentiabilities
"""
        )

        for op_name, entries in sorted(by_names.items(), key=lambda kv: kv[0]):
            fs.write(f"""

differentiabilities[torch.{op_name}] = differentiabilities[torch.ops.aten.{op_name}] = ["""
            )

            for entry in entries:
                fs.write("""
    Differentiability(
        ["""
                )
                for diff_param in entry.differentiability.diffable_params:
                    fs.write(f"'{diff_param}', ")
                fs.write(
        "],"
                )

                fs.write("""
        ["""
                )
                for other_param, default in entry.differentiability.other_params:
                    if isinstance(default, RequiredParam):
                        default_str = 'required'
                    else:
                        default_str = repr(default)
                    fs.write(f"('{other_param}', {default_str}), ")
                fs.write(
        "],"
                )
            
                fs.write("""
    ),""")

            fs.write("""
]""")




op_names = [
    'abs',
    'add',
    'add_',
    'clone',
    'concat',
    'div',
    'exp',
    'lt',
    'mul',
    'neg',
    # 'pow',
    'sign',
    'sub',
    'sub_',
    'sum',
    'sort',
    'where',
]


@dataclasses.dataclass
class CliArgs:
    pytorch_codebase: str

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--pytorch_codebase', type=str)

    cliargs = CliArgs(**vars(parser.parse_args()))

    result_entries = parse_derivatives_yaml(cliargs, op_names)
    dump_torch_jvp_differentiabilities_file(result_entries)