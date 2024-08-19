# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7:27

RUN apt-get update && apt-get install -y \
    git \
    zip

WORKDIR /app/
RUN wget https://developer.download.nvidia.com/devtools/nsight-systems/NsightSystems-linux-cli-public-2023.4.1.97-3355750.deb
RUN apt install ./NsightSystems-linux-cli-public-2023.4.1.97-3355750.deb

# These pip packages have some issues and must be installed separately.
RUN pip install Cython mpi4py

RUN pip install \
	more-itertools \
	numpy \
	scipy \
	matplotlib \
	nvtx \
	mgmetis

# For launching and attaching to TensorBoard sessions within the container
RUN pip install tensorboard torch-tb-profiler