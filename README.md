<div align="center">
  <img width="500px" src="tutorial/easier logo.png"/>
</div>

---

# What is EASIER?

Manufacturing engineers and natural science researchers have long struggled with ad-hoc implementations of physical simulations and manual deployment on large clusters, preventing them from fully focusing on their products and scientific research.
**EASIER** (**E**fficient **A**uto-scalable **S**cientific **I**nfrastructure for **E**ngineers and **R**esearchers) is a domain specific language for

, just-in-time compiler, and high-performance runtime to make development and deployment of physical simulations as easy as that of large language models.



# Getting Started

Manufacturing engineers and natural science researchers have long struggled with ad-hoc implementations of physical simulations and manual deployment on large-scale clusters, which prevents them from fully focusing on their products and research. **EASIER** (**E**fficient **A**uto-scalable **S**cientific **I**nfrastructure for **E**ngineers and **R**esearchers) is a domain-specific language, compiler, and runtime that simplifies the development and deployment of physical simulations, similar to large language models. This saves R&D costs for industries and research labs.

scientific computing tasks and their hand-craft deployment on clusters.


 efficiently and automatically scaling scientific computing tasks up and out, providing scientific computing developers similar experience to that of developing and deploying large-scale deep learning models.

## Setup the development environment

The dependencies should be fixed to exactly the acceptable minimun versions to
ensure the widest compatibility and development consistency.

The Python itself should be fixed to 3.8:

```shell
# for conda
conda create -n ENV_NAME python=3.8
conda activate ENV_NAME

# for other venvs
# TODO
```

To install the dependencies for development:

```shell
pip install Cython mpi4py               # must be installed separately
pip install -r dev-requirements.txt
pip install -e .                        # equals `python setup.py develop`
```

## Project folder structure
```bash
├── docker/       # Dockerfiles
├── easier/       # python package
│   ├── core/     # jit compiler implementation
│   └── **/**     # numerical algorithms based on eaiser jit compiler
├── tests/        # unit tests
├── dev-requirements.txt
├── README.md
├── setup.py
└── .gitignore
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
