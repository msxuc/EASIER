# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import operator
from types import ModuleType
from typing import Callable, Dict, Iterator, List, Tuple
from typing_extensions import Literal
import more_itertools
import os

import torch
from torch import nn
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.proxy import Proxy
from torch.fx.graph import Graph
from easier.core.runtime.dist_env import set_runtime_dist_env_backend

import easier as esr
from easier.core import passes
from easier.core import module as _EsrMod
from easier.core.utils import EasierJitException


class EasierProxy(Proxy):
    """Custom Proxy to trace additional operations like `__setitem__`.
    """

    def __init__(self, node: Node, tracer: 'EasierTracer'):
        super().__init__(node, tracer)

    def __getattr__(self, name):
        """
        Let Tensor method calls result in `Node[op='call_method']`.

        But forbid tracing-time access to Tensor attribute like `shape, ndim`,
        otherwise it results in
        `Node[op='call_function',target=operator.getattr]`,
        which requires constant propagation process to analyze.
        """

        # Prompt common errors.
        if name in ['shape', 'size', 'dim', 'ndim', 'numel', 'nelement']:
            raise NotImplementedError(
                "Currently EASIER does not support accessing common"
                " hyperparameter information '" + name + "' during tracing."
                " If needed, those values must be calculated ahead-of-time.")

        # This will result in insertion `Node[op='call_method',target='xxx']`
        # into the graph.
        # NOTE the `Node.target` attribute is a string of the method name.
        return super().__getattr__(name)

    def __setitem__(self, indices, value) -> Proxy:
        return self.tracer.create_proxy(
            'call_function', operator.setitem,
            (self, indices, value), {})

    # NOTE
    # Besides __getattr__ and __setitem__:
    # - __getitem__ has been handled by base Proxy class and results in
    #   Node{op='call_function', target=operator.getitem:callable}
    #   see
    #   https://github.com/pytorch/pytorch/blob/v1.13.0/torch/fx/proxy.py#L395
    #   https://github.com/pytorch/pytorch/blob/v1.13.0/torch/fx/graph.py#L1473


class EasierTracer(Tracer):
    """Custom Tracer to label easier atomic modules and functions as leaf nodes
    """

    def __init__(self,
                 autowrap_modules: Tuple[ModuleType, ...] = (),
                 autowrap_functions: Tuple[Callable, ...] = (),
                 param_shapes_constant: bool = False) -> None:
        super().__init__(
            # NOTE both esr and _EsrMod modules are required, so that
            # invocations like `esr.sum` and `_EsrMod.sum` are hooked by FX.
            tuple(autowrap_modules) + (math, esr, _EsrMod),  # type: ignore
            autowrap_functions, param_shapes_constant)

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # register modules in easier.core.ops as leaf module during tracing
        return (super().is_leaf_module(m, module_qualified_name)
                or m.__module__.startswith("easier.core.module"))

    def proxy(self, node: Node) -> EasierProxy:
        return EasierProxy(node, tracer=self)


def infer_and_enforce_unique_device_type(modules: List[esr.Module]) -> str:
    rec_sub_items: Dict[torch.Tensor, str] = {}

    def _update(named_items: Iterator[Tuple[str, torch.Tensor]]):
        for name, item in named_items:
            rec_sub_items[item] = name  # name may overwrite, but ok for debug.

    for mi, m in enumerate(modules):
        prefix = \
            f"<{m.__class__.__module__}.{m.__class__.__name__}" \
            f" at modules[{mi}]>"
        _update(m.named_parameters(prefix=prefix, recurse=True))
        _update(m.named_buffers(prefix=prefix, recurse=True))

    device_type_grouped: Dict[str, str] \
        = more_itertools.map_reduce(
            rec_sub_items.items(),
            keyfunc=lambda kv: kv[0].device.type,
            valuefunc=lambda kv: kv[1],
            reducefunc=lambda name_list: name_list[0])

    if len(device_type_grouped) != 1:
        bad_items = ', '.join(
            f'{prefixed_name} on {dev}'
            for dev, prefixed_name in device_type_grouped.items())
        raise EasierJitException(
            "Must involve only one torch device type (cpu/cuda)."
            f" At least {bad_items} have incompatible devices.")

    return more_itertools.first(device_type_grouped.keys())


def _enforce_device_type_cpu_cuda(device_type: str) -> Literal['cuda', 'cpu']:
    # TODO new codegen backends that have no match communication backend
    # require further design here.
    if device_type not in ['cpu', 'cuda']:
        raise EasierJitException(f'device type {device_type} not cpu or cuda')
    return device_type  # type: ignore


def _fully_load_data(top_modules: List[esr.Module]):
    """
    Fully load index and data, onto the initial device of the data loader.
    """
    for root in top_modules:
        for m in root.modules():  # recursively
            if isinstance(m, (esr.Selector, esr.Reducer)):
                if not m.easier_index_ready:
                    m.idx = m.easier_data_loader.fully_load(device=None)
                    m.easier_index_ready = True

        for p in root.parameters(recurse=True):
            if isinstance(p, esr.Tensor):
                if not p.easier_data_ready:
                    p.data = p.easier_data_loader.fully_load(device=None)
                    p.easier_data_ready = True


def _validate_nonjit_state(top_modules: List[esr.Module]):
    def _raise():
        raise EasierJitException("Input easier.Modules have been compiled.")

    for root in top_modules:
        for m in root.modules():  # recursively
            if isinstance(m, (esr.Selector, esr.Reducer)):
                if m.easier_index_ready:
                    _raise()

        for p in root.parameters(recurse=True):
            if isinstance(p, esr.Tensor):
                if p.easier_data_ready:
                    _raise()


def compile(modules: List[esr.Module],  # type: ignore
            backend: Literal['torch', 'cpu', 'gpu', 'none', None] = None):
    """Just in time compilation for a list of fx compatible torch.nn.Module

    Args:
        modules (List[nn.Module]): the list of torch modules to be jitted
        backend (str): backend platform that modules should be compiled to,
            supporting:
            - "torch": inherit the device specified by `modules`
            - "gpu": enforce CUDA for now
            TODO will support AMD too, need to check if torch/dist can
                transparently switch to AMD infrastructure when adding support.
            - "cpu": CPU
            - "none": disable jit and return `modules` directly
            - None: use the value specified by environment variable
                EASIER_COMPILE_BACKEND.
                If EASIER_COMPILE_BACKEND is not defined, use default backend
                "torch".

            Please note the difference between string-typed value `"none"` and
            object-typed value `None`.

    Returns:
        GraphModule: the jitted module that can run on the specified backend
            platform distributively
    """
    if backend is None:
        # Python keyword `None` is different from string "none":
        # - `None` means the compile backend is not specified at all at the
        #   invocation of `compile()`, the backend to use will be decided by
        #   a chain of rules;
        # - "none" means the JIT compilation is turned off.
        #   "none" may be a result decided by the rules when `backend is None`.

        # The env var "EASIER_COMPILE_BACKEND" may be set by EASIER Launcher
        # command line argument `--backend`.
        env_backend = os.environ.get("EASIER_COMPILE_BACKEND", None)

        if env_backend is None:
            backend = 'torch'
        elif env_backend in ['torch', 'cpu', 'gpu', 'none']:
            backend = env_backend  # type: ignore
        else:
            raise EasierJitException(
                "Detected invalid value of EASIER_COMPILE_BACKEND: "
                + env_backend
            )
    assert backend is not None

    # Retrieve and validate esr.Modules as inputs, even backend==none.
    top_modules = modules
    _validate_nonjit_state(top_modules)

    loading_flags = \
        set(top_m.easier_loaded_cache is None for top_m in top_modules)
    if len(loading_flags) != 1:
        raise EasierJitException(
            "easier.Modules to compile must be all loaded or all not loaded")

    modules: List[esr.Module] = []
    for module in top_modules:
        if not isinstance(module, esr.Module):
            raise EasierJitException(
                f"Instance of {module.__class__} cannot be jitted")
        for m in module.modules():
            if m in modules:
                continue
            if isinstance(m, esr.Module):
                modules.append(m)

    for m in modules:
        m.easier_jit_backend = backend

    # No matter what backend is specified, we enforce the input modules are
    # on the same device, like 'cuda:3'.
    # And specifically for CUDA, the device ID will be ignored, only the
    # _device type_ 'cuda' will be kept, and the distribution pass will scatter
    # tensors to other devices like `cuda:0, cuda:1, cuda:2` etc.
    orig_device_type = infer_and_enforce_unique_device_type(modules)

    if backend == 'none':
        esr.logger.info("EASIER just-in-time compilation is turned off")
        esr.logger.info(
            "Any HDF5 dataset to initialize easier.Tensor/Selector/Reducer"
            " will be fully loaded")

        _fully_load_data(top_modules)
        return top_modules

    elif backend == 'torch':
        comm_backend = _enforce_device_type_cpu_cuda(orig_device_type)
    elif backend == 'gpu':
        comm_backend = 'cuda'  # TODO enforce GPU == CUDA for now
    elif backend == 'cpu':
        comm_backend = backend
    else:
        raise EasierJitException(f"Argument `jit_backend` cannot be {backend}")

    esr.logger.info(
        f"EASIER just-in-time compilation has started, backend={backend}")

    set_runtime_dist_env_backend(comm_backend)

    tracer = EasierTracer()
    graphs: List[Graph] = [tracer.trace(m) for m in modules]

    # passes
    modules, graphs = passes.propagate_metadata(modules, graphs)
    modules, graphs = passes.group_tensors(modules, graphs)

    modules, graphs = passes.distribute_dataflow(modules, graphs)

    graph_modules: List[GraphModule] = []
    for m, g in zip(modules, graphs):
        graph_modules.append(GraphModule(m, g))

    for m, gm in zip(modules, graph_modules):
        m.forward = gm.forward

    esr.logger.info("EASIER just-in-time compilation has completed")

    return top_modules
