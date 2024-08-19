# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime
import os
import random
import string
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Sequence
import argparse
import shlex

import torch.distributed as dist
from easier.core.utils import logger, get_random_str


class LocalAgentArgs(argparse.Namespace):
    # path to the env vars dump, e.g. /tmp/easier/erun/tmpabcdef_node-K.env
    env_vars_dump: str
    node_addr: str

    nnodes: int
    node_rank: int
    nproc_per_node: int
    master_port: int
    easier_compile_backend: str

    user_program: str
    user_program_args: Sequence[str]


def run_local_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_vars_dump", type=str
    )
    parser.add_argument(
        "--node_addr", type=str
    )
    #
    # Some important env vars that EAISER programs should see.
    # The names may be duplicated with torchrun/LocalAgent env vars, but the
    # meanings are very different.
    #
    parser.add_argument(
        "--nnodes", type=int
    )

    # torchrun doesn't officially set NODE_RANK
    parser.add_argument(
        "--node_rank", type=int
    )
    parser.add_argument(
        "--nproc_per_node", type=int
    )

    # We can read master_addr from LocalAgent env vars.
    parser.add_argument(
        "--master_port", type=int
    )

    parser.add_argument(
        "--easier_compile_backend", type=str
    )

    # Positional arg
    parser.add_argument(
        "user_program", type=str
    )
    parser.add_argument(
        "user_program_args", nargs=argparse.REMAINDER
    )
    args = parser.parse_args(namespace=LocalAgentArgs())

    # The master_port in env to init this process group is the "--launcher_port"
    # specified in the EASIER launcher, different from the env var "MASTER_PORT"
    # to init process group of the NCCL comm backend in the EASIER programs.
    dist.init_process_group('gloo',
                            timeout=datetime.timedelta(minutes=5))

    if args.node_rank == 0:
        session_id = get_random_str()
        dist.broadcast_object_list([session_id])
    else:
        _objs = [None]
        dist.broadcast_object_list(_objs)
        session_id = _objs[0]  # type: ignore

    # Internally managed session ID
    session_id: str

    nprocs: List[int] = [None for _ in range(args.nnodes)]  # type: ignore
    dist.all_gather_object(nprocs, args.nproc_per_node)

    addrs: List[str] = [None for _ in range(args.nnodes)]  # type: ignore
    dist.all_gather_object(addrs, args.node_addr)

    world_size: int = sum(nprocs)

    with open(
        os.path.join(os.path.dirname(__file__), "setup_template.sh")
    ) as template_f:
        template_lines = template_f.readlines()

    erun_tmp_dir = os.path.join(tempfile.gettempdir(), 'easier', 'erun')
    os.makedirs(erun_tmp_dir, exist_ok=True)

    # e.g. /tmp/easier/erun/a9b8c7d6.sh
    sess_sh = os.path.join(erun_tmp_dir, session_id + '.sh')

    with open(sess_sh, mode='w+') as sh_f:
        for tp_line in template_lines:

            if tp_line.startswith('## @@ LOCAL ENVIRONMENT VARIABLES @@ ##'):
                # This script `local_agent.py` will be launched by
                # a `nproc=1` torchrun instance,
                # so here we already have some torchrun env vars,
                # and those are likely NOT correct for EASIER processes.
                #
                # So we dump the env vars before launching local agent,
                # and pass them in via a temp file at `env_vars_dump`.
                #
                # NOTE unlike `mpirun_wrapper.sh` which always read env vars
                # on node-0, the script here read env vars on arbitrary node
                # it's launched, so we need to hardcode the .env path into it.
                sh_f.writelines([
                    f'source {args.env_vars_dump}\n',
                ])

                master_addr = os.environ["MASTER_ADDR"]  # same as LocalAgent

                # Some other important env vars the EAISER processes may see:
                sh_f.writelines([
                    f'export MASTER_ADDR="{master_addr}"\n',
                    f'export MASTER_PORT="{args.master_port}"\n',
                    'export EASIER_COMPILE_BACKEND='
                    + args.easier_compile_backend + '\n',
                ])

            else:
                sh_f.write(tp_line)

    # ensure scripts are ready on all nodes
    dist.barrier()

    # We are executing mpirun on rank-(0 of nnodes) LocalAgent,
    # where we would have a process tree:
    # ```
    # - shell-on-node-0
    #   |-  torchrun-on-node-0
    #       |-  rank-0-LocalAgent
    #           |-  mpirun
    #               |-  mpi-daemon
    #                   |-  rank-(0 of nnodes*nproc)-EASIER-program
    #                   |-  rank-(1 of nnodes*nproc)-EASIER-program
    #                   |-  ...
    # ```
    # rank-0-LocalAgent (which is, here) will have torchrun-specific env vars
    # like NODE_RANK, RANK etc., and those env vars will be inherit by mpirun,
    # a child process.
    # We specifically want mpirun to have only the env vars that are in
    # shell-on-node-0.
    # Otherwise, any possible `torch.dist.init_process_group` in the innermost
    # EASIER programs may conflict with `init_process_group` in the LocalAgent,
    # causing mpirun to hang.
    #
    # We rely on Linux utilities:
    # - `env -i EXE ARGS`
    #   run cmd `EXE ARGS` in a pure new environment, inheriting no env vars
    #   from the parent process;
    # - `bash --noprofile --norc SH ARGS`
    #   run script `SH ARGS` without loading system-level or user-level bash
    #   profiles, resulting in no new env vars defined.
    #
    # The overall command looks like:
    # ```
    # env -i bash --noprofile --norc \
    # easier_install/launcher/mpirun_wrapper.sh \
    #       /tmp/easier/erun/tmpRANDOM_node-0.env \
    #       mpirun -np ... -host ... -bind-to ... \
    #           env -i bash --noprofile --norc \
    #           /tmp/easier/erun/SESS_ID.sh \
    #               python program.py ARGS
    # ```

    if args.node_rank == 0:
        # e.g. "node-0:8,node-1:12,node-2:16"
        mpirun_host = ",".join(
            f"{addr}:{nproc}" for addr, nproc in zip(addrs, nprocs)
        )

        subp_args = [
            "env", "-i",
            "bash", "--noprofile", "--norc",
            os.path.join(os.path.dirname(__file__), "mpirun_wrapper.sh"),
            args.env_vars_dump,

            "mpirun",
            "-np", str(world_size),
            "-host", shlex.quote(mpirun_host),

            # TODO need a better way to configure core affinity (as well as
            # OMP_NUM_THREADS, now we do it in the .sh)
            "--bind-to", "core",

            # don't load any user bash profile, as we have dumped all env vars
            # in the shell invoking EASIER Launcher.
            "bash", "--noprofile", "--norc",
            sess_sh,

            "python", args.user_program
        ]

        subp_args.extend(args.user_program_args)

        logger.info(' '.join(subp_args))

        # TODO if this LocalAgent process is abnormally terminated
        # (e.g. killed, KeyboardInterrupt-ed), the mpirun processes will keep
        # going and become children of `/init`.
        mpirun_p = subprocess.Popen(subp_args)  # non-blocking

        max_interval = 30

        while True:

            for _ in range(max_interval):
                # locally poll every second
                time.sleep(1)
                exitcode: Optional[int] = mpirun_p.poll()
                finish: bool = exitcode is not None

                # as soon as mpirun ends, continue to notify other nodes
                if finish:
                    break

            # No matter what the exitcode is, send True to signal the finish
            dist.broadcast_object_list([finish], src=0)

            if finish:
                return

    else:
        while True:
            objs: List[object] = [None]

            # Given the maximum timeout is 5 mins (set during init_process),
            # this collective call will block until rank-0 LocalAgent queries
            # the exitcode of mpirun.
            # Only rank-0 needs to deal with the time interval, as long as
            # it doesn't exceed 5 mins or causing TimeoutError.
            dist.broadcast_object_list(objs, src=0)

            if objs[0] == True:
                return

    # TODO we may delete the sess_sh in a collectively call too


if __name__ == '__main__':
    run_local_agent()
