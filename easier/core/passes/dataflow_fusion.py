# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import operator
from typing import Union
from collections.abc import Iterable

from torch import fx, nn

import easier as esr
from easier.core.passes.metadata_propagation import Role
from easier.core.runtime.modules import HaloExchanger
from easier.core.utils import tree_map, OrderedSet


def get_submodule(root: nn.Module, node: fx.Node):
    assert node.op == "call_module", \
        "Can only get submodule from call_moduel node."
    submod_path, _sep, attr_name = node.target.rpartition(".")
    submod = root.get_submodule(submod_path)
    return getattr(submod, attr_name)


class NodeGroup:
    def __init__(self, root: nn.Module, node: fx.Node):
        self.root = root

        self.nodes = OrderedSet()
        self.args = OrderedSet()
        self.users = OrderedSet()
        self.control_args = OrderedSet()
        self.control_users = OrderedSet()

        self.append(node)
        self.upstream(node)
        self.downstream(node)

        self.nodes = list(self.nodes)
        self.nodes = sorted(self.nodes, key=lambda n: n.meta['easier_index'])

    def is_instance(self, node: fx.Node, cls: nn.Module) -> bool:
        if node.op == "call_module":
            target = get_submodule(self.root, node)
            if isinstance(target, cls):
                return True
        return False

    def append(self, node: fx.Node):
        node.meta['easier_visit'] = True
        self.nodes.add(node)
        self.args.discard(node)
        self.users.discard(node)
        self.control_args.discard(node)
        self.control_users.discard(node)

        for arg in node.all_input_nodes:
            if isinstance(arg, fx.Node) and arg not in self.nodes:
                self.args.add(arg)

        for arg in node.meta['control_args']:
            if arg not in self.nodes:
                self.control_args.add(arg)

        for usr in node.users:
            if usr not in self.nodes:
                self.users.add(usr)

        for usr in node.meta['control_users']:
            if usr not in self.nodes:
                self.control_users.add(usr)

    def upstream(self, node: fx.Node) -> None:
        if self.is_instance(node, esr.Gather):
            return
        elif self.is_instance(node, HaloExchanger):
            return
        elif node.op == 'output':
            return
        elif isinstance(node.meta['easier_meta'], Iterable):
            # assume only local operter produces iterable meta
            return

        arg_list = list(node.all_input_nodes) + node.meta['control_args']
        for arg in arg_list:
            if not isinstance(arg, fx.Node):
                continue
            elif arg.meta['easier_visit']:
                continue
            elif arg.meta['easier_wait']:
                continue
            elif arg.op == "get_attr":
                continue
            elif self.is_instance(arg, esr.Scatter):
                continue
            elif arg.meta['easier_meta'].role == Role.REPLICA:
                continue
            elif arg.meta['easier_meta'].role == Role.EDGE:
                self.append(arg)
                self.upstream(arg)
                self.downstream(arg)
                continue

            is_cycle = False
            garg_list = list(self.args) + list(self.control_args)
            for garg in garg_list:
                usr_list = list(arg.users) + arg.meta['control_users']
                for auser in usr_list:
                    if auser not in self.nodes and \
                            auser.meta['easier_index'] in \
                            garg.meta['easier_upstream']:
                        is_cycle = True
                        break
                if is_cycle:
                    break
            if is_cycle:
                continue

            self.append(arg)
            self.upstream(arg)
            self.downstream(arg)

    def downstream(self, node: fx.Node) -> None:
        if self.is_instance(node, esr.Scatter):
            if 'easier_scatter_out' in node.meta.keys():
                for out in node.meta['easier_scatter_out']:
                    if out.meta['easier_wait']:
                        self.append(out)
                        # unlock all users as soon as this Scatter take one of
                        # them.
                        for out in node.meta['easier_scatter_out']:
                            out.meta['easier_wait'] = False
                        break
            return
        elif node.op == "get_attr":
            return
        elif isinstance(node.meta['easier_meta'], Iterable):
            return
        elif node.meta['easier_meta'].role == Role.REPLICA:
            return
        elif self.is_instance(node, HaloExchanger):
            return

        user_list = list(node.users) + node.meta['control_users']
        for usr in user_list:
            if not isinstance(usr, fx.Node):
                continue
            elif usr.meta['easier_visit']:
                continue
            elif usr.meta['easier_wait']:
                continue
            elif usr.op == "output":
                continue
            elif self.is_instance(usr, esr.Gather):
                continue
            elif self.is_instance(usr, HaloExchanger):
                continue
            elif usr.meta['easier_meta'].role == Role.EDGE or \
                    self.is_instance(usr, esr.Scatter):
                self.append(usr)
                self.upstream(usr)
                self.downstream(usr)
                continue

            is_cycle = False
            gusr_list = list(self.users) + list(self.control_users)
            for gusr in gusr_list:
                gusr_idx = gusr.meta['easier_index']
                if gusr_idx != usr.meta['easier_index'] and \
                        gusr_idx in usr.meta['easier_upstream']:
                    is_cycle = True
                    break
            if is_cycle:
                continue

            self.append(usr)
            self.upstream(usr)
            self.downstream(usr)

    def build(self) -> Union[fx.Node, fx.GraphModule]:
        if len(self.nodes) == 1:
            # get_attr, output, local nodes, HaloExchanger
            return self.nodes[0]

        node_dict = {}
        new_graph = fx.Graph()
        for arg in self.args:
            n = new_graph.create_node('placeholder', arg.name)
            n.meta['easier_meta'] = arg.meta['easier_meta']
            node_dict[arg.name] = n

        # When not accessing via `.all_input_nodes`, operation like `stack`
        # may have nested input e.g.
        # Node{target=torch.stack, args=([a1,a2,...], -1)}
        def _try_get_node_dict(arg):
            if isinstance(arg, fx.Node):
                return node_dict[arg.name]
            else:
                return arg

        for n in self.nodes:
            args = []
            for arg in n.args:
                args.append(tree_map(arg, _try_get_node_dict))

            kwargs = {}
            for key, value in n.kwargs.items():
                kwargs[key] = tree_map(value, _try_get_node_dict)

            node = new_graph.create_node(
                n.op, n.target, tuple(args), kwargs, n.name)
            node.meta['easier_meta'] = n.meta['easier_meta']
            node_dict[n.name] = node

        output = []
        meta = []
        for n in self.users:
            for arg in n.all_input_nodes:
                if isinstance(arg, fx.Node) and \
                        arg.name in node_dict.keys() and \
                        node_dict[arg.name].op != "placeholder" and \
                        node_dict[arg.name] not in output:
                    output.append(node_dict[arg.name])
                    meta.append(arg.meta['easier_meta'])

        node = new_graph.create_node(
            'output', 'output', args=(output,), name='output')
        node.meta['easier_meta'] = meta

        return fx.GraphModule(self.root, new_graph)


def _fuse(root: nn.Module, graph: fx.Graph):
    # setup metadata for fusion
    for i, node in enumerate(graph.nodes):
        if 'easier_meta' not in node.meta.keys():
            raise esr.EasierJitException(
                'Dataflow fusion must run after metadata propagation.')
        node.meta['control_args'] = []
        node.meta['control_users'] = []
        node.meta['easier_visit'] = False
        node.meta['easier_wait'] = False
        node.meta['easier_index'] = i
        node.meta['easier_upstream'] = set([i])

    for node in graph.nodes:
        # setup control flow dependency for fusion
        if node.meta['easier_is_inplace'] is not None:
            for user in node.meta['easier_is_inplace'].users:

                if user.meta['easier_index'] < node.meta['easier_index']:
                    if node not in user.meta['control_users']:
                        user.meta['control_users'].append(node)
                    if user not in node.meta['control_args']:
                        node.meta['control_args'].append(user)

                if user.meta['easier_index'] > node.meta['easier_index']:
                    if node not in user.meta['control_args']:
                        user.meta['control_args'].append(node)
                    if user not in node.meta['control_users']:
                        node.meta['control_users'].append(user)

        # handle scatter op that is not full
        if node.op == "call_module":
            m = get_submodule(root, node)
            if isinstance(m, esr.Scatter) and not m.is_full:
                node.meta['easier_scatter_out'] = []
                for usr in node.users:
                    if not (usr.target is esr.norm) and \
                       not (usr.target is esr.sum) and \
                       not (usr.target is esr.mean):
                        node.meta['easier_scatter_out'].append(usr)
                        usr.meta['easier_wait'] = True

    for i, node in enumerate(graph.nodes):
        for arg in node.all_input_nodes:
            node.meta['easier_upstream'] = \
                node.meta['easier_upstream'].union(
                    arg.meta['easier_upstream'])

        for arg in node.meta['control_args']:
            node.meta['easier_upstream'] = \
                node.meta['easier_upstream'].union(
                    arg.meta['easier_upstream'])

    # group nodes
    node_groups = []
    for node in graph.nodes:
        if node.meta['easier_visit']:
            continue
        elif node.meta['easier_wait']:
            continue
        else:
            node_groups.append(NodeGroup(root, node))

    def _topo_cmp(a: NodeGroup, b: NodeGroup):
        for n in a.nodes:
            for arg in b.args:
                if n.meta['easier_index'] in arg.meta['easier_upstream']:
                    return -1

        for n in b.nodes:
            for arg in a.args:
                if n.meta['easier_index'] in arg.meta['easier_upstream']:
                    return 1

        return 0

    def _topo_sort(node_groups):
        new_node_groups = []
        while len(node_groups):
            a = node_groups[0]
            for b in node_groups:
                if _topo_cmp(a, b) > 0:
                    a = b
            new_node_groups.append(a)
            node_groups.remove(a)
        return new_node_groups

    # topo sort node groups
    node_groups = _topo_sort(node_groups)

    # build fused graph
    num = 0
    node_dict = {}

    def _try_get_node_dict(arg):
        if isinstance(arg, fx.Node):
            if arg.name in node_dict.keys():
                return node_dict[arg.name]
            else:
                assert False, \
                    'A required arg node is not in the node list, ' \
                    'probably caused by incorrect topological order'
        else:
            return arg

    new_graph = fx.Graph()
    for ng in node_groups:
        gm = ng.build()

        if isinstance(gm, fx.Node):
            args = []
            for arg in gm.args:
                args.append(tree_map(arg, _try_get_node_dict))

            kwargs = {}
            for key, value in gm.kwargs.items():
                kwargs[key] = tree_map(value, _try_get_node_dict)

            node = new_graph.create_node(
                gm.op, gm.target, tuple(args), kwargs, gm.name)
            node.meta['easier_meta'] = gm.meta['easier_meta']
            node_dict[node.name] = node

        elif isinstance(gm, fx.GraphModule):
            model_name = f'easier{num:04d}'
            num += 1

            args = []
            for arg in ng.args:
                if isinstance(arg, fx.Node):
                    if arg.name in node_dict.keys():
                        args.append(node_dict[arg.name])
                    else:
                        assert False, \
                            f'A required arg node {arg.name}' \
                            ' is not in the node list, ' \
                            'probably caused by incorrect topological order'
                else:
                    assert False, \
                        f'Node group {model_name} has an argument' \
                        ' that is not fx.Node.'

            root.add_module(model_name, gm)
            call_module_node = new_graph.create_node(
                "call_module", model_name, tuple(args), name=model_name)
            output = list(gm.graph.nodes)[-1]
            call_module_node.meta['easier_meta'] = output.meta['easier_meta']

            for i, arg in enumerate(output.args[0]):
                node = new_graph.create_node(
                    "call_function",
                    operator.getitem,
                    (call_module_node, i), name=arg.name)
                node.meta['easier_meta'] = arg.meta['easier_meta']
                node_dict[node.name] = node
        else:
            assert False, \
                f'A node group {model_name} has an argument' \
                ' that is not fx.Node.'

    return root, new_graph


def fuse_dataflow(modules, graphs):
    new_modules = []
    new_graphs = []
    for m, g in zip(modules, graphs):
        new_m, new_g = _fuse(m, g)
        new_modules.append(new_m)
        new_graphs.append(new_g)

    return new_modules, new_graphs
