<div align="center">
  <img width="400px" src="tutorial/logo.png"/>
</div>

# What is EASIER?

**EASIER** is a domain specific language embedded in PyTorch to automatically scale physical simulations up and out.
It just-in-time (JIT) distributes tensor dataflows that describe physical simulations to any number of workers and compiles them for any kind of accelerators without requiring any code changes.
This makes the development of high-performance physical simulations on explosively growing AI supercomputers is as easy as that of large language models.

# Get Started

### Installation

To ensure the compatibility of all dependencies, the Python verion should be fixed to 3.8 for now.

```shell
# for conda
conda create -n ENV_NAME python=3.8
conda activate ENV_NAME

# for conda, run this only when you see compile/link errors with following `pip` commands
conda install gxx_linux-64
```

Clone the repo and install EASIER as well as all dependencies:

```shell
git clone https://github.com/microsoft/EASIER.git
cd EASIER

# for Ubuntu
sudo apt-get install libopenmpi-dev

pip install Cython==3.0.11 mpi4py==3.1.5    # must be installed separately
pip install -r dev-requirements.txt
pip install -e .                            # equals `python setup.py develop`
```

### Run examples

Solve shallow water equations in a square box:
```shell
# create folder to store simulation results
mkdir res

# launch the simulation in a `torchrun` style
easierrun --nnodes=1 --nproc_per_node=4 tutorial/shallow_water_equation.py --backend=cpu --output=res
```

EASIER launcher `easierrun` provides similar user experience to that of `torchrun`.
Just add `--master_addr` argument to the launcher if there are multiple nodes.

Plot the simulation results
```shell
python tutorial/swe_plot.py --data_dir res --filename swe.gif
```

<div align="center">
  <img width="500px" src="tutorial/swe.gif"/>
</div>

# Tutorial
Detailed tutorial and more examples are coming soon.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
