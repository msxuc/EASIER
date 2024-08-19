# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .metadata import (
    ScalarType, promote_scalar_types,
    Role, EasierTensorMeta, get_node_meta, StructuredTensorMeta,
    get_meta_from_ir_literal,
    convert_scalar_type_to_torch_dtype,
    convert_torch_dtype_to_scalar_type
)

from .rule_registry import (
    MetadataRuleBase, metadata_rule_registry, inplace_version,
)

from .metadata_propagation import (
    propagate_metadata
)

#
# Activate all metadata propagation rules for PyTorch frontend
#
from . import torch_frontend
