# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script will be launched by `env -i bash --noprofile --norc`
# (`env -i`: do not inherit from parent; 
#  `bash --noprofile --norc`: do not read bash profiles)
# therefore we have no preexisting env vars here.
#
# This script will be invoked with arguments:
# ```
# env -i bash --noprofile --norc \
# easier_install/launcher/mpirun_wrapper.sh \
#       /tmp/easier/erun/tmpRANDOM_node-0.env \
#       mpirun -np ... -host ... -bind-to ... \
#           env -i bash --noprofile --norc \
#           /tmp/easier/erun/SESS_ID.sh \
#               program.py ARGS
# ```

# $1 is the path to `node-0.env` which records env vars in the invoking shell,
# by `source`-ing it we re-define those env vars in the current `env -i` shell.
# 
# And NOTE, it's always `node-0.env` that's `source`-ed, because we are running
# `mpirun` on node-0.
# We simply re-define env vars, without introducing `torchrun`-specific
# env vars, also, without merging possibly different env vars from all nodes.
source $1
shift

# Till here,
# $@ are the arguments `mpirun -np ... ... ...` like above.
eval "$@"