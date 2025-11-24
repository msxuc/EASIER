# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# import pushforward

# import torch_ops_manual
# import torch_ops_generated

"""
NOTE torch_opinfo_generated should not be imported here, as auto-generation
of autodiff rules has these steps:

1.  `gen_autodiff` first generates `torch_ops_generated` module;
2.  this `easier.core.autodiff` module is imported for `gen_autodiff.py` to
    read all @pushforward definition (both in `_manual` and in `_generated`);
3.  `gen_autodiff.py` generates `torch_opinfo_generated` module, which is only
    used by user-coding-time calls to EASIER autodiff APIs like `jvp`.
"""
# DO NOT
# import torch_opinfo_generated
# DO NOT