# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Environment variables existing before invoking EASIER Launcher,
# will be injected by EASIER Launcher's LocalAgent. E.g.,
# ```
# # redefine env vars from the erun-invoking shell:
# source /tmp/easier/erun/tmpRANDOM_node-K.env
# ...
# # injected by the LocalAgent:
# export MASTER_ADDR=node-0
# export MASTER_PORT=29500
# export EASIER_COMPILE_BACKEND=torch
# ````
## @@ LOCAL ENVIRONMENT VARIABLES @@ ##

# Signature of use EASIER launcher
export EASIER_USE_EASIER_LAUNCHER=1

# MPI-assigned env vars may override default env vars, such as NODE_RANK.
export NODE_RANK=$OMPI_COMM_WORLD_NODE_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_WORLD_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK

# TODO to avoid oversubscribing cores, refine this into e.g. 
# ```
# if LOCAL_WORLD_SIZE > 1 and OMP_NUM_THREADS not in os.environ:
#   log.warning("suggest configure OMP_NUM_THREADS")
#   OMP_NUM_THREADS = 1
# ```
export OMP_NUM_THREADS=${OMP_NUM_THREADS:=1}

# TODO `cd` to the erun-invoking bash `pwd`

# Arguments to this script look like:
# `python program.py --ARGS`
# and the `python` EXE is prepended by the LocalAgent.
eval "$@"
