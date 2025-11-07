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
class Param:
    type: str
    name: Optional[str]
    
    # NOTE `p=None` in yaml results in default="None" -- a str, not Python nil
    default: Optional[str]

    kind: Literal['positional', 'keyword', 'return']

@dataclasses.dataclass
class OpDef:
    # name is without overloading suffix
    name: str

    # may be empty
    overloading_suffix: str

    # e.g. (Tensor self, int param) -> Tensor
    func_type: str

    # e.g. (Tensor self, int param)
    param_list: str

    # e.g. Tensor
    return_type: str

    named_params: List[Param]

    # certain ops do not have results, generally should be excluded in EASIER
    named_return: Union[Param, List[Param], None]

    

    # decision: Literal['']

    # variants: List[Literal['function', 'method']]

    def get_name_with_suffix(self):
        if self.overloading_suffix:
            return f'{self.name}.{self.overloading_suffix}'
        else:
            return self.name
    
    def __repr__(self) -> str:
        return f'{self.get_name_with_suffix()}{self.func_type}'
    
    def __hash__(self) -> int:
        return hash(self.get_name_with_suffix())

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

def _parse_op_def(funcsig: str) -> OpDef:
    # Extreme cases are like:
    # split.Tensor(Tensor(a -> *) self, SymInt split_size, \
    #       int dim=0) -> Tensor(a)[]
    func_type_str = '(' + '('.join(funcsig.split('(')[1:])
    ast_level = 0
    for i, c in enumerate(func_type_str):
        if c == '(':
            # First char in func_type is always '(' so AST level begins
            # with 1.
            ast_level += 1
        if c == ')':
            ast_level -= 1
        
        if ast_level == 0:
            break
    param_list_str = func_type_str[:(i+1)]
    assert param_list_str[0] == '('
    assert param_list_str[-1] == ')'

    from_arrow = func_type_str[(i+1):].strip()
    assert from_arrow.startswith('-> ')
    return_type = from_arrow[len('-> '):]

    name_w_suffix = funcsig.split('(')[0]
    name_wo_suffix = name_w_suffix.split('.')[0]
    overloading_suffix = name_w_suffix[(len(name_wo_suffix)+1):]

    params = param_list_str[1:-1].split(', ')
    named_params: List[Param] = []

    param_kind = 'positional'

    for param in params:

        if param == '*':
            param_kind = 'keyword'
            continue

        defaults = param.split('=')
        assert len(defaults) <= 2

        if len(defaults) == 2:
            default = defaults[1]  # maybe str "None"
        else:
            default = None  # Python nil object None
        
        type_name = defaults[0]
        segs = type_name.split(' ')
        param_name = segs[-1]
        param_type = ' '.join(segs[:-1])  # e.g 
        named_param = Param(param_type, param_name, default, param_kind)
        named_params.append(named_param)

    if return_type == '()':
        named_return = None
    
    elif return_type.startswith('('):
        return_items = return_type[1:-1].split(', ')

        named_return = []
        for return_item in return_items:
            assert '=' not in return_item
            segs = return_item.split(' ')
            assert len(segs) <= 2
            if len(segs) == 2:
                item_name = segs[1]
            else:
                item_name = None
            item_type = segs[0]
            named_item = Param(item_type, item_name, None, 'return')
            named_return.append(named_item)


    else:
        segs = return_type.split(' ')
        assert len(segs) <= 2
        if len(segs) == 2:
            item_name = segs[1]
        else:
            item_name = None
        item_type = segs[0]
        named_return = Param(item_type, item_name, None, 'return')

    return OpDef(
        name=name_wo_suffix,
        overloading_suffix=overloading_suffix,
        func_type=func_type_str,
        param_list=param_list_str,
        return_type=return_type,
        named_params=named_params,
        named_return=named_return
    )

# Get all public, traceable, non-backprop operator definitions, e.g.
# "all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"
def parse_native_functions_yaml(args: 'CliArgs') -> Tuple[List[OpDef], Set[OpDef]]:
    nf_yaml_fp = os.path.join(
        args.pytorch_codebase, 'aten/src/ATen/native/native_functions.yaml'
    )

    with open(nf_yaml_fp, 'r') as yaml_fs:
        funcdefs: list = yaml.safe_load(yaml_fs)

    opdefs: List[OpDef] = []

    # Collect all public, traceable ops
    for funcdef in funcdefs:
        funcdef: dict

        funcsig: str = funcdef['func']

        # torch internal operators, skip, e.g.
        # "_unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor"
        if funcsig.startswith('_'):
            continue

        opdef = _parse_op_def(funcsig)

        # Exclude backprop-only ops
        if opdef.name.endswith('_backward'):
            continue

        # Exclude Tensor-creators like zeros/eye etc. as they don't get traced
        # or appear on FX Graph.
        #
        # However, some Tensor-creators may have `out=` parameter e.g.
        # eye.m_out(SymInt n, SymInt m, *, Tensor(a!) out) -> Tensor(a!)
        if 'Tensor' not in opdef.param_list:
            continue

        opdefs.append(opdef)
    
    # quick lookup table
    key=lambda d: d.name
    opdefs_by_name: Dict[str, List[OpDef]] = {
        k: list(v) for k, v in
        itertools.groupby(sorted(opdefs, key=key), key=key)
    }


    # Filter out some op defs
    removed_inplace_ops: Set[OpDef] = set()
    for inp_opdef in opdefs:
        #
        # For inplace ops, if there exist non-inplace versions and the only
        # difference is '_'-in-name or `out`-parameter, such kind of inplace
        # ops can be handled by EASIER like their non-inplace versions plus
        # setitem.
        #
        if inp_opdef.name.endswith('_'):
            # The 1st param is inplace, e.g.
            # add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) \
            #   -> Tensor(a!)
            noninplace_name = inp_opdef.name[:-1]

        elif '!' in inp_opdef.param_list:
            # Generally the last param(s) is inplace, e.g.
            # addmv.out(Tensor self, Tensor mat, Tensor vec, *, \
            #   Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
            noninplace_name = inp_opdef.name

            # When check the non-inplace version, we should note that
            # the inplace version has an extra '*' delimiter in the params.

        else:
            continue

        if noninplace_name in opdefs_by_name:
            noninplace_versions = opdefs_by_name[noninplace_name]

            #
            # If a sequence of terms `Tensor(x!) param` appear in the parameter
            # list and in the result, too, we treat this inplace operator as
            # SIMPLE inplace op;
            # and if a counterpart with the same parameter list, but
            # not inplace, we treat that inplace op as INCREMENTAL inplace op.
            #
            params = inp_opdef.param_list[1:-1].split(', ')

            def _validate_param(param: str):
                segs = param.split('=')
                assert len(segs) <= 2  # if have defaults

                # not expecting cases like `Tensor(x!)? p=None` that's both
                # inplace and optional.
                assert not ('!' in param and '?' in param)  

                assert ',' not in segs[0]

            for p in params:
                _validate_param(p)

            #
            # Find out `out` parameters
            #
            # If a `out=` parameter is a tuple/list, it's like
            # split_copy.Tensor_out(Tensor self, SymInt split_size, \
            #   int dim=0, *, Tensor(a!)[] out) -> ()
            #                 ~~~~~~~~~~~~
            # If multiple parameters are inplace, they may not be named as
            # `out` in their definitions (but their Python APIs will have `out`
            # parameter of a tuple of Tensors):
            # cummax.out(Tensor self, int dim, *, \
            #   Tensor(a!) values, Tensor(b!) indices) \
            #   -> (Tensor(a!) values, Tensor(b!) indices)

            inp_params: List[Tuple[int, str]] = []
            prev_inp_param_pos = -1
            for i, p in enumerate(params):
                # e.g. Tensor(a!) Tensor(b!) Tensor(c!)
                inp_id = chr(ord('a') + len(inp_params))
                inp_tensor_type = f'Tensor({inp_id}!)'
                if p.startswith(inp_tensor_type):
                    inp_params.append((i, p))
                
                    # If multiple inplace args, they must be sequential. 
                    if prev_inp_param_pos >= 0:
                        assert i == prev_inp_param_pos + 1
                    prev_inp_param_pos = i
                

            # Check if inplace func type is symmetric:
            # (Tensor(a!) arg) -> Tensor(a!)
            #
            # Modify the inplace part in the param list to get the result type.
            # The convention is, the result of an inplace op must be also the
            # inplace parameters.
            #
            if len(inp_params) == 1:
                is_symmetric_inp_op = inp_opdef.return_type == 'Tensor(a!)'
                noninp_res_type = 'Tensor'
            else:
                # e.g. ['Tensor(a!)', 'Tensor(b!)']
                #
                # Some ops do not follow the convention e.g.
                # max.dim_max(Tensor self, int dim, bool keepdim=False, *, \
                #   Tensor(a!) max, Tensor(b!) max_values) -> ( \
                #   Tensor(a!) values, Tensor(b!) indices)
                # But the result item names take no effect in resolution.
                if not (
                    inp_opdef.return_type[0] == '(' \
                        or inp_opdef.return_type[-1] == ')'
                ):
                    is_symmetric_inp_op = False
                    noninp_res_type = 'WHATEVER'

                else:
                    # Ignore result item names, only check if type lists match.
                    is_symmetric_inp_op = \
                        list(
                            p.split(' ')[0] for i, p in inp_params
                        ) == list(
                            p.split(' ')[0] for p
                            in inp_opdef.return_type[1:-1].split(', ')
                        )

                    noninp_res_type = re.sub(
                        '\\(\\w!\\)', '', inp_opdef.return_type
                    )


            if is_symmetric_inp_op:
                #
                # Remove (a!) for memory alias -- inplace target -- in the
                # param/result types
                #
                if inp_opdef.name.endswith('_'):
                    # Expect the inplace param to be the 1st
                    assert prev_inp_param_pos == 0
                    assert len(inp_params) == 1

                    noninp_params = list(params)
                    for i, p in inp_params:
                        noninp_params[i] = re.sub('\\(\\w!\\)', '', p)
                
                else:
                    # Inplace version has keyword `out` param, we should remove
                    # keyword delimiter * in the param list.
                    assert len(inp_params) >= 1

                    noninp_params = list(params)
                    for maxi, p in sorted(inp_params, key=lambda ip: ip[0], reverse=True):
                        noninp_params.pop(maxi)
                    if noninp_params[-1] == '*':
                        noninp_params.pop()

                noninp_func_type = '('  + ', '.join(noninp_params) + ') -> ' + noninp_res_type


                for noninp in noninplace_versions:
                    if noninp.func_type == noninp_func_type:

                        removed_inplace_ops.add(inp_opdef)

                        break
                else:
                    print(
                        f'{inp_opdef}\n\tis not an INCREMENTAL inplace op'
                    )
            else:
                # An extra kind of inplace ops: non-symmetric
                # where the inplace arguments are not returned.
                print(
                    f'{inp_opdef}\n\tdoes not have a SYMMETRIC noninplace'
                    ' version'
                )

        else:  # !if noninplace_name in opdefs_by_name:
            print(
                f'{inp_opdef}\n\tdoes not have noninplace version'
            )
        

    opdefs = list(filter(lambda d: d not in removed_inplace_ops, opdefs))
    
    return opdefs, removed_inplace_ops


TangentField: TypeAlias = Literal['RESULT', 'RESULT_N', 'NAMED_RESULT', 'UNKNOWN']

def _parse_derivative_expression2(opdef: OpDef, field: TangentField, auto: bool, cpp_expr: str):
    vars: List[str] = re.findall(R'([A-Za-z][A-Za-z0-9_:]*)', cpp_expr)
    assert len(vars) > 0

    op_derived_terms = set(['grad'])
    # for named_param in opdef.named_params

    return vars


def parse_derivatives_yaml(
    args: 'CliArgs', opdefs: List[OpDef], removed_incremental_inplace_ops: Set[OpDef]
) -> Tuple[List[DerivativeDefinition], List[ExceptionalDefinition]]:
    print("""
##############################
#   Parse derivatives.yaml   #
##############################
""")

    derivatives_yaml_fp = os.path.join(
        args.pytorch_codebase, 'tools/autograd/derivatives.yaml'
    )

    # TODO some op result item has alias like Q K V, and does not have 'result' field
    with open(derivatives_yaml_fp, 'r') as yaml_fs:
        torch_derivatives: list = yaml.safe_load(yaml_fs)
    
    opdef_by_sig = { 
        f'{opdef.get_name_with_suffix()}{opdef.func_type}': opdef
        for opdef in opdefs
    }
    removed_incr_inp_by_sig = {
        f'{opdef.get_name_with_suffix()}{opdef.func_type}': opdef
        for opdef in removed_incremental_inplace_ops
    }

    _audit_non_diffable_input_types: Dict[str, List[OpDef]] = {}

    _vars = set()

    _simple_results = set()
    _auto = set()

    for yaml_derivdef in torch_derivatives:
        yaml_derivdef: dict

        deriv_op_sig = cast(str, yaml_derivdef['name'])

        # torch internal operators
        if deriv_op_sig.startswith('_'):
            continue

        deriv_opdef = _parse_op_def(deriv_op_sig)

        if deriv_opdef.name.endswith('_backward'):
            continue

        if 'dispatch' in yaml_derivdef:
            yaml_derivdef = yaml_derivdef['dispatch']['Default']

        field: TangentField
        if 'result' in yaml_derivdef:
            field = 'RESULT'
        elif 'result0' in yaml_derivdef:
            field = 'RESULT_N'
        elif isinstance(deriv_opdef.named_return, list) and any(
                named_res_item.name in yaml_derivdef
                for named_res_item in deriv_opdef.named_return
            ):
            field = 'NAMED_RESULT'
        else:
            field = 'UNKNOWN'  # matmul alike



        if deriv_op_sig in removed_incr_inp_by_sig:
            assert field == 'RESULT' or field == 'UNKNOWN'
            #
            # Why derivatives.yaml still has rules for INCREMENTAL inplace ops?
            # Check if there's anything special
            #
            if field == 'RESULT':
                tangent_expr: str = yaml_derivdef['result']
                if tangent_expr in ['auto_element_wise', 'auto_linear', 'self_t.zero_()']:
                    continue
                    # else: default cases, follow EASIER rules:
                    # 1) calculate tangent as non-inplace, 2) setitem.

                else:
                    print(
                        f'{deriv_op_sig}\n\tinplace but does not have TRIVIAL result tangent:'
                        f'\n\t{tangent_expr}\n'
                    )
                    continue

            else:
                print(
                    f'{deriv_op_sig}\n\tinplace but does not have "result" tangent defined'
                    f'\n\t{yaml_derivdef}\n'
                )
                continue
        
        else:

            # not for inplace ops
            if field == 'RESULT':
                tangent_expr: str = yaml_derivdef['result']
                auto = tangent_expr in ['auto_element_wise', 'auto_linear']

                if auto:
                    if 'self' not in yaml_derivdef:
                        print(
                            f'{deriv_op_sig}\n\tis auto but does not have "self" tangent defined'
                            f'\n\t{yaml_derivdef}\n'
                        )
                        continue
                    
                    else:
                        tangent_expr = yaml_derivdef['self']
                
                else:
                    pass

                tangent_expr = tangent_expr.strip()
                vars = _parse_derivative_expression2(deriv_opdef, field, auto, tangent_expr)
            
            elif field == 'RESULT_N':
                assert isinstance(deriv_opdef.named_return, list)

                for i in range(len(deriv_opdef.named_return)):

                    if f'result{i}' not in yaml_derivdef:
                        # Not used but check it's consistent
                        if 'output_differentiability' in yaml_derivdef:
                            output_diffables: List[bool] = yaml_derivdef['output_differentiability']
                            assert output_diffables[i] == False
                        # Assert no mixed use of resultN and result item name
                        assert deriv_opdef.named_return[i].name not in yaml_derivdef
                        continue

                    tangent_expr_i: str = yaml_derivdef[f'result{i}']
                    tangent_expr_i = tangent_expr_i.strip()

                    assert not tangent_expr_i.startswith('auto')
                    auto = False

                    vars = _parse_derivative_expression2(deriv_opdef, field, auto, tangent_expr_i)
                
            else:

                # Check output_differentiabilty
                if 'output_differentiability' in yaml_derivdef:
                    output_diffables: List[bool] = yaml_derivdef['output_differentiability']
                    if isinstance(deriv_opdef.named_return, list):
                        assert len(output_diffables) == len(deriv_opdef.named_return)
                    else:
                        assert len(output_diffables) == 1

                    if not any(output_diffables):
                        # No output propagate tangents
                        # print(
                        #     f'{deriv_op_sig}\n\tdoes not have differentiable output'
                        #     f'\n\t{yaml_derivdef}\n'
                        # )
                        continue

                # Check if input is marked non-diff-able, or its type is effectively non-diff-able
                # NOTE this does not fully identify non-diff-able inputs
                input_marked_as_non_diffables = []
                special_type_nondiffable_input = False
                for p in deriv_opdef.named_params:
                    if 'Tensor' not in p.type:
                        _audit_non_diffable_input_types.setdefault(p.type, []).append(deriv_opdef)
                        special_type_nondiffable_input = True
                        input_marked_as_non_diffables.append(True)
                    else:
                        input_marked_as_non_diffables.append(
                            yaml_derivdef.get(p.name, "") == "non_differentiable"
                        )
                if all(input_marked_as_non_diffables):
                    # No input propagate tangents

                    # if special_type_nondiffable_input:
                    #     # Only warn to ensure `'Tensor' not in` assumption.
                    #     print(
                    #         f'{deriv_op_sig}\n\tdoes not have differentiable input (w/ special input type)'
                    #         f'\n\t{yaml_derivdef}\n'
                    #     )

                    continue
                
                # field in [RESULT, RESULT_N] are handled before
                for k in yaml_derivdef.keys():
                    assert not k.startswith('result')
                
                # If an input is still diffable and handled, its field in yaml
                # is its name.
                inputs_handled = [False] * len(deriv_opdef.named_params)
                for i, (marked_non_diffable, p) in enumerate(
                    more_itertools.zip_equal(
                        input_marked_as_non_diffables, deriv_opdef.named_params
                    )
                ):
                    if marked_non_diffable:
                        continue

                    if p.name in yaml_derivdef:
                        inputs_handled[i] = True
                
                if all(
                    nondiff or handled for nondiff, handled
                    in more_itertools.zip_equal(
                        input_marked_as_non_diffables, inputs_handled
                    )
                ):
                    continue
            
                else:

                    print(
                        f'{deriv_op_sig}\n\ttangent unknown'
                        f'\n\t{yaml_derivdef}\n'
                    )
                    continue
        
        # if result_tangent in ['auto_element_wise', 'auto_linear']:
        #     if 'self' in derivdef:
        #         self_expr = derivdef['self'].strip()
        #         vars = _parse_derivative_expression2(deriv_opdef, self_expr)
        #         for var in vars:
        #             if 'jvp' not in var and 'backward' not in var:
        #                 _vars.add(var)
        #                 _auto.add(deriv_opdef)
                
        # # endif in removed_incr_inp
        # if result_tangent not in ['NOT_RESULT', 'auto_element_wise', 'auto_linear']:
        #     result_tangent = result_tangent.strip()
        #     vars = _parse_derivative_expression2(deriv_opdef, result_tangent)
        #     for var in vars:
        #         if 'jvp' not in var and 'backward' not in var:
        #             _vars.add(var)
        #             _simple_results.add(deriv_opdef)

            # else:
                # print(
                #     f'{deriv_op_sig}\n\tParse succeeded'
                #     f'\n\tRAW: {result_tangent}\n\tRES: {ast}\n'
                # )

    print('Nondiffable input types:', list(_audit_non_diffable_input_types.keys()))

    print(sorted(_vars))

    print(len(_auto))
    print(len(_simple_results))
        


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

    cliargs = CliArgs(**vars(parser.parse_args()))

    # if args.mode == 'collect_op_names':
    if True:
        """
        The `names.yaml` is important because it marks what operators are
        known by EASIER: they are either differentiable or not.

        With the assumption that PyTorch never changes the op names and the
        overloading suffixes, the `names.yaml` incrementally grows as new
        PyTorch versions come out.

        If an op not in `names.yaml` is used, EASIER cannot be sure it's
        differentiable or not, so autodiff process will be interrupted and
        the user can define a custom derivative rule as a temporary solution.
        """
        opdefs, removed_inplace_ops = parse_native_functions_yaml(cliargs)

        op_names_fp = os.path.join(os.path.dirname(__file__), 'names.yaml')
        with open(op_names_fp, 'w') as op_names_fs:
            yaml.safe_dump(
                list(map(dataclasses.asdict, opdefs)),
                op_names_fs,
                sort_keys=False,
                width=float("inf")
            )

    
        parse_derivatives_yaml(cliargs, opdefs, removed_inplace_ops)