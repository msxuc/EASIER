# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
EASIER launcher

Steps:

1.  All EASIER launchers that are collectively executed under `torchrun`.
    These EASIER launchers determine an random _EASIER session ID_.

2.  From the _interactive shell_ on node-K (0 <= K < WORLD_SIZE) that
    EASIER launcher is executed, extract all environment variables,
    which may be configured by `.bashrc` etc.

    Save the environment variables to `/tmp/easier/erun/SESSION_ID.sh`
    which is the same path on all nodes.

3.  On node-0, run
    ```
    mpirun -host HOST -np WORLD_SIZE /tmp/easier/erun/SESSION_ID.sh \
        user_program.py user_args ...
    ```

    NOTE
    We already have behaviors:
    -   When one `mpirun`-launched process dies,
        all `mpirun`-launched processes die, including `mpirun` itself.
    -   When the `mpirun` process itself dies or exits,
        the EAISER launcher on node-0 then exits.
    -   When any EASIER launcher dies or exits, all other EASIER launchers on
        other nodes (they are collectively launched by `torchrun`) exit too.

    We need to further ensure:
    -   On node-0, when the EASIER launcher is killed,
        terminate all `mpirun` processes.
        (otherwise, for example, they will be moved as children of `/init`).

        For example, `subprocess.run()`-launched subprocess will:
        -   accept Python KeyboardInterrupt and exit;
        -   not be aware of Python gets `kill -9`, become a child of `/init`;
        -   TODO aware of torchrun cancelled because of signal from other node?
"""


import argparse
import os
import sys
from typing import Optional, Sequence
import tempfile
import shlex
from typing_extensions import Literal

import torch.distributed.run as DR


class LauncherArgs(argparse.Namespace):
    # NOTE the collection of console args is a less-structured data class.

    nnodes: int
    node_rank: int
    nproc_per_node: int

    master_addr: str
    launcher_port: int
    master_port: int
    node_addr: Optional[str]

    backend: Literal['torch', 'gpu', 'cpu', 'none']

    log_dir: Optional[str]
    session_id: Optional[str]

    user_program: str
    user_program_args: Sequence[str]


def parse_arg():
    parser = argparse.ArgumentParser()

    # TODO torchrun accepts ['auto', 'cpu', 'gpu', int]
    parser.add_argument(
        "--nproc_per_node", type=int,
        help="The number of workers on this node"  # may be different on each
    )

    parser.add_argument(
        "--nnodes", type=int, default=1,
        help="The number of nodes"
    )
    parser.add_argument(
        "--node_rank", type=int, default=0,
        help="The rank of this node"
    )

    parser.add_argument(
        "--master_addr", type=str, default="127.0.0.1",
        help="The address of the master node (rank-0 node)"
    )
    parser.add_argument(
        "--launcher_port", type=int, default=59500,
        help="The port for EASIER launcher. The address-port pair"
        " master_addr:launcher_port will be bound."
        " This port should be different from --master_port."
    )

    # LocalAgent needs to call `torch.dist.init_process_group()` with
    # "master_addr:master_port" once.
    #
    # The addr:port pair may
    # only usable when the comm backend is NCCL/GLOO which would be initialized
    # by torch.distributed.init_process_group that needs a master_port.
    parser.add_argument(
        "--master_port", type=int, default=29500,
        help="The port for PyTorch NCCL communication backend, if used."
        " The address-port pair"
        " master_addr:launcher_port will be bound."
        " This port should be different from --launcher_port."
    )

    parser.add_argument(
        "--node_addr", type=str, default=None,
        help="The address of this node for `mpirun` to access."
        " By default the hostname of this node will be used."
    )

    parser.add_argument(
        "--backend", type=str,
        choices=['torch', 'gpu', 'cpu', 'none'], default='torch',
        help="The compilation backend the EASIER programs will target."
        " The backend can be overridden by the `backend` argument of"
        " `easier.compile()` function in the code."
    )

    # Simply forward to underlying torchrun.
    # Due to mpirun, all stdout/stderr will be merged to rank-0/node-0
    # and written to log_dir there.
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="The dir to save redirected stdout/stderr stream."
        " By default, erun does not save these contents."
    )
    # Human-readable session ID/name to save logs
    parser.add_argument(
        "--session_id", type=str, default=None,
        help="The session name (suggested to be human-readable)"
        " of the distributed task."
    )

    # Positional arg
    parser.add_argument(
        "user_program", type=str
    )
    parser.add_argument(
        "user_program_args", nargs=argparse.REMAINDER
    )

    args = parser.parse_args(namespace=LauncherArgs())

    return args


def main():
    # the `main()` function is entrypoint to the console script `erun`,
    # it's specified in `/setup.py` and `erun` is automatically created by PIP.
    #
    # Basically, running
    # `erun --nproc_per_node=N --ERUN_ARGS program.py PROG_ARGS`
    # equals to
    # ```
    # torchrun --nproc_per_node=1 --ERUN_ARGS \
    #   easier_install\launcher\local_agent.py \
    #   --nproc_per_node=N \
    #   program.py PROG_ARGS
    # ```
    args = parse_arg()

    erun_tmp_dir = os.path.join(tempfile.gettempdir(), 'easier', 'erun')
    os.makedirs(erun_tmp_dir, exist_ok=True)

    # NOTE we specifically capture env vars here, instead of in the LocalAgent,
    # because the LocalAgent will be launched by torchrun, therefore it will
    # see (or, be polluted with) many torchrun-specific env vars, like RANK.
    # And more importantly, the distribution configuration of LocalAgent
    # is different than the configuration of the real EASIER process,
    # e.g. in the LocalAgent, `nproc_per_node = 1` but in EASIER processes
    # we generally have `nproc_per_node > 1`.
    #
    # e.g. write to /tmp/easier/erun/tmpRANDOM_node-3.env
    with tempfile.NamedTemporaryFile(
        mode='w', delete=False,
        dir=erun_tmp_dir,
        suffix=f'_node-{args.node_rank}.env'
    ) as env_f:
        for k, v in os.environ.items():
            quoted_v = shlex.quote(v)
            env_f.write(f'export {k}={quoted_v}\n')

    if args.node_addr is None:
        # os.uname returns (sysname, nodename, release, ...)
        node_addr = os.uname()[1]  # type: ignore
    else:
        node_addr = args.node_addr

    local_agent = os.path.join(os.path.dirname(__file__), "local_agent.py")

    # Arguments to the torchrun that launches LocalAgent
    localagent_torchrun_args = [
        "--nnodes", str(args.nnodes),
        "--node_rank", str(args.node_rank),
        "--nproc_per_node", "1",

        "--master_addr", args.master_addr,
        "--master_port", str(args.launcher_port)
    ]

    # TODO torchrun log_dir has torchrun-specific, nontrivial structure:
    # e.g. `LOGDIR/RDZVID_RANDOM/attempt_0/RANK/stderr.log`
    # By reusing torchrun log_dir argument we add the torchrun specificity
    # to EASIER launcher.
    if args.log_dir is not None:
        localagent_torchrun_args.extend([
            "--log_dir", args.log_dir,
            "--tee", "3"  # redirect and duplicate stdout and stderr
        ])

    if args.session_id is not None:
        localagent_torchrun_args.extend([
            "--rdzv_id", args.session_id
        ])

    # Arguments to the LocalAgent itself
    localagent_torchrun_args.extend([
        local_agent,
        "--env_vars_dump", env_f.name,
        "--node_addr", node_addr,

        "--nnodes", str(args.nnodes),
        "--node_rank", str(args.node_rank),
        "--nproc_per_node", str(args.nproc_per_node),
        "--master_port", str(args.master_port),
        "--easier_compile_backend", args.backend,
    ])

    localagent_torchrun_args.append(args.user_program)
    localagent_torchrun_args.extend(args.user_program_args)

    DR.main(localagent_torchrun_args)
