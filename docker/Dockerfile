# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7:27

RUN apt-get update && apt-get install -y \
    git \
    zip

WORKDIR /app/
RUN wget https://developer.download.nvidia.com/devtools/nsight-systems/NsightSystems-linux-cli-public-2023.4.1.97-3355750.deb
RUN apt install ./NsightSystems-linux-cli-public-2023.4.1.97-3355750.deb

RUN pip install \
	more-itertools==10.4.0 \
	numpy==1.24.4 \
	scipy==1.10.1 \
	matplotlib==3.7.5 \
	networkx==3.1

# WARNING
# mgmetis wrongly specifies Cython a "install time" requirement and mpi4py
# as optional but they should both be "setup time" requirements.
# Therefore we need to install them separatedly before
# `pip install mgmetis` or `pip install -r dev-requirements.txt`
# so that ParMETIS support can be added.
RUN pip install \
	Cython==3.0.11 \
	mpi4py==3.1.5

RUN pip install \
	mgmetis==0.1.1 \
	h5py==3.11.0

# To profile using PyTorch integration of tensorboard
# Both packages need to be installed on the machine where TensorBoard will be
# launched.
RUN pip install \
	tensorboard \
	torch-tb-profiler

# To profile using Nsight Systems
RUN pip install \
	nvtx
