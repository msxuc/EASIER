# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import dataclasses
import itertools
import os
import re
from typing import Dict, List, Literal, Optional, Set, Tuple, TypedDict, Union
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

    # decision: Literal['']

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


# Get all public, traceable, non-backprop operator definitions, e.g.
# "all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor"
def parse_native_functions_yaml(args: 'CliArgs') -> List[OpDef]:
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

        # Exclude backprop-only ops
        if name_wo_suffix.endswith('_backward'):
            continue

        # Exclude Tensor-creators like zeros/eye etc. as they don't get traced
        # or appear on FX Graph.
        #
        # However, some Tensor-creators may have `out=` parameter e.g.
        # eye.m_out(SymInt n, SymInt m, *, Tensor(a!) out) -> Tensor(a!)
        if 'Tensor' not in param_list_str:
            continue

        opdefs.append(OpDef(
            name=name_wo_suffix,
            overloading_suffix=overloading_suffix,
            func_type=func_type_str,
            param_list=param_list_str,
            return_type=return_type
        ))
    
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






            # TODO split _ and out cases, if out, out if always keyword params
            # when checking incrementalness, append a * if none.






        elif '!' in inp_opdef.param_list:
            # Generally the last param(s) is inplace, e.g.
            # addmv.out(Tensor self, Tensor mat, Tensor vec, *, \
            #   Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
            noninplace_name = inp_opdef.name
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
                

            # Expect the inplace param to be the 1st
            if inp_opdef.name.endswith('_'):
                assert prev_inp_param_pos == 0
                assert len(inp_params) == 1
            

            assert len(inp_params) > 0

            if len(inp_params) == 1:
                simple_return_type = 'Tensor(a!)'
            else:
                simple_return_type = \
                    '(' + ', '.join(p for i, p in inp_params) + ')'


            if inp_opdef.return_type == simple_return_type:

                noninp_params = list(params)
                for i, p in inp_params:
                    noninp_params[i] = re.sub('(\\w!)', '', p)
                
                noninp_return_type = re.sub('(\\w!)', '', simple_return_type)

                noninp_func_type = '('  + ', '.join(noninp_params) + ') -> ' + noninp_return_type


                for noninp in noninplace_versions:
                    if noninp.func_type == noninp_func_type:

                        removed_inplace_ops.add(inp_opdef)

                        break
                else:
                    print(
                        f'{inp_opdef}\n\tis not an INCREMENTAL inplace op'
                    )
            else:
                print(
                    f'{inp_opdef}\n\tdoes not have a SIMPLE noninplace'
                    ' version'
                )

        else:  # !if noninplace_name in opdefs_by_name:
            print(
                f'{inp_opdef}\n\tdoes not have noninplace version'
            )
        

    opdefs = list(filter(lambda d: d not in removed_inplace_ops, opdefs))
    
    return opdefs

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
        op_sigs = parse_native_functions_yaml(args)
        # ops = []
        # for func in op_sigs:
        #     op = OpDef(func=func)
        #     ops.append(op)
        
        # op_names_fp = os.path.join(os.path.dirname(__file__), 'names.yaml')
        # with open(op_names_fp, 'w') as op_names_fs:
        #     yaml.safe_dump(ops, op_names_fs)

    