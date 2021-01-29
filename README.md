# pyribs

|                     Source                     |                                                       PyPI                                                        |                                                                                                                  CI/CD                                                                                                                   |                        Docs                        |                                                                   Docs Status                                                                    |
| :--------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
| [GitHub](https://github.com/icaros-usc/pyribs) | [![PyPI](https://img.shields.io/pypi/v/ribs.svg?style=flat-square&color=blue)](https://pypi.python.org/pypi/ribs) | [![Automated Testing](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Ficaros-usc%2Fpyribs%2Fbadge&style=flat-square)](https://github.com/icaros-usc/pyribs/actions?query=workflow%3A"Automated+Testing") | [ribs.readthedocs.io](https://ribs.readthedocs.io) | [![Documentation Status](https://readthedocs.org/projects/ribs/badge/?version=latest&style=flat-square)](https://readthedocs.org/projects/ribs/) |

A _bare-bones_ quality diversity optimization library. Our library is the official implementation of the Covariance Matrix Adaptation MAP-Elites algorithm and implements the _Rapid Illumination of Behavior Spaces (RIBS)_ redesign of MAP-Elites detailed in the paper [Covariance Matrix Adapation for the Rapid Illumination of Behavior Space](https://arxiv.org/abs/1912.02400).

## Overview

Unlike traditional optimizers which seek to find a single high-performing
solution to a problem, Quality-Diversity (QD) algorithms seek to discover
multiple high-performing solutions. These solutions are characterized by
properties known as behavior characteristics (BCs). After a single run, a QD
algorithm outputs an archive with the solutions it has found. Each solution is
the highest-performing one in a certain region of the behavior space. pyribs
follows the Rapid Illumination of Behavior Spaces framework introduced in
[Fontaine 2020](https://arxiv.org/abs/1912.02400). Under this framework, pyribs
divides a QD algorithm into three components:

- The **Archive** stores solutions found by the algorithm so far.
- **Emitters** (one or more) take the archive and decide how to generate new
  solutions from it.
- An **Optimizer** joins the algorithm together. The optimizer repeatedly
  generates solutions from the archive using the emitters, and adds the
  evaluated solutions back into the archive.

## Usage Example

pyribs uses an ask-tell interface similar to that of
[pycma](https://pypi.org/project/cma/). The following example shows how to run
the RIBS version of MAP-Elites. Specifically, we create:

- A 2D **GridArchive** with 20 bins and a range of (-1, 1) in each dimension.
- A **GaussianEmitter**, which in this case starts by drawing examples from a
  Gaussian distribution centered at **0** with standard deviation 0.1. After the
  first iteration, this emitter selects random solutions in the archive and adds
  Gaussian noise to it with standard deviation 0.1.
- An **Optimizer** that combines the archive and emitter together.

After creating the components, we then run on the negative 10-D Sphere function
for 1000 iterations. To keep our BCs simple, we use the first two entries of
each 10D solution vector as our BCs.

```python
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer

archive = GridArchive([20, 20], [(-1, 1), (-1, 1)])
emitters = [GaussianEmitter(archive, [0.0] * 10, 0.1)]
optimizer = Optimizer(archive, emitters)

for itr in range(1000):
    solutions = optimizer.ask()

    objectives = -np.sum(np.square(solutions), axis=1)
    bcs = solutions[:,:2]

    optimizer.tell(objectives, bcs)
```

To visualize this archive, we can then use Seaborn's `heatmap` like so:

```python
import seaborn as sns

data = archive.as_pandas().pivot('index-0', 'index-1', 'objective')
sns.heatmap(data)
```

![Sphere heatmap](readme_assets/sphere_heatmap.png)

For more information, please refer to the
[documentation](https://ribs.readthedocs.io/).

## Installation

pyribs supports Python 3.6 and greater. Earlier versions may work but are not
officially supported.

To install from PyPI, run

```bash
pip install ribs
```

This command only installs dependencies for the core of pyribs. To be able to use
tools like `ribs.visualize`, run

```bash
pip install ribs[all]
```

To install a version from source, clone this repo, cd into it, and run

```bash
pip install -e .[all]
```

To test your installation, import it and print the version with:

```bash
python -c "import ribs; print(ribs.__version__)"
```

You should see a version number like `0.2.0` in the output.

## Documentation

See here for the documentation: <https://ribs.readthedocs.io>

To serve the documentation locally, clone the repo and install the development
requirements with

```bash
pip install -e .[dev]
```

Then run

```bash
make servedocs
```

This will open a window in your browser with the documentation automatically
loaded. Furthermore, every time you make changes to the documentation, the
preview will also reload.

## Contributors

This project was completed in the [ICAROS Lab](http://icaros.usc.edu) at USC.

- [Bryon Tjanaka](https://btjanaka.net)
- [Matt Fontaine](https://github.com/tehqin)
- [Yulun Zhang](https://github.com/lunjohnzhang)
- [Sam Sommerer](https://github.com/sam-som-usc)
- Nikitas Klapsis
- [Stefanos Nikolaidis](https://stefanosnikolaidis.net)

## License

pyribs is released under the
[MIT License](https://github.com/icaros-usc/pyribs/blob/master/LICENSE).

## Credits

This package was initially created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
