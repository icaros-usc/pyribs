# pyribs

|                     Source                     |                                                       PyPI                                                        |                                                                                                                  CI/CD                                                                                                                   |                        Docs                        |                                                                   Docs Status                                                                    |
| :--------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
| [GitHub](https://github.com/icaros-usc/pyribs) | [![PyPI](https://img.shields.io/pypi/v/ribs.svg?style=flat-square&color=blue)](https://pypi.python.org/pypi/ribs) | [![Automated Testing](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Ficaros-usc%2Fpyribs%2Fbadge&style=flat-square)](https://github.com/icaros-usc/pyribs/actions?query=workflow%3A"Automated+Testing") | [ribs.readthedocs.io](https://ribs.readthedocs.io) | [![Documentation Status](https://readthedocs.org/projects/ribs/badge/?version=latest&style=flat-square)](https://readthedocs.org/projects/ribs/) |

A _bare-bones_ quality diversity optimization library. The pyribs library is the
official implementation of the Covariance Matrix Adaptation MAP-Elites (CMA-ME)
algorithm and implements the _Rapid Illumination of Behavior Spaces (RIBS)_
redesign of MAP-Elites detailed in the paper
[Covariance Matrix Adapation for the Rapid Illumination of Behavior Space](https://arxiv.org/abs/1912.02400).

## Overview

![Types of Optimization](readme_assets/optimization_types.png)

[Quality-diversity (QD) optimization](https://arxiv.org/abs/2012.04322) is a
subfield of optimization where solutions generated cover every point in a
behavior space while simultaneously maximizing (or minimizing) a single
objective. QD algorithms within the MAP-Elites family of QD algorithms produce
heatmaps (archives) as output where each cell contains the best discovered
representative of a region in behavior space.

While many QD libraries exist, this particular library aims to be the QD analog
to the [pycma](https://pypi.org/project/cma/) library (a single objective
optimization library). In contrast to other QD libraries, this library is
"bare-bones" meaning pyribs (like [pycma](https://pypi.org/project/cma/))
focuses solely on optimizing fixed dimensional continuous domains. Focusing
solely on this one commonly-occurring problem allows us to optimize the library
for performance as well as simplicity of use. For applications of QD on discrete
domains, we recommend using [qdpy](https://gitlab.com/leo.cazenille/qdpy/) or
[sferes](https://github.com/sferes2/sferes2).

A user of pyribs selects three components that meet the needs of their
application:

- An **Archive** saves the best representatives generated within behavior space.
- **Emitters** control how new candidate solutions are generated and effect if
  the algorithm prioritizes quality or diversity.
- An **Optimizer** joins the **Archive** and **Emitters** together and acts as a
  scheduling algorithm for emitters. The **Optimizer** provides an iterface for
  requesting new candidate solutions and telling the algorithm how candidates
  performed.

## Usage

Here we show an example application of CMA-ME in pyribs. To initialize the
algorithm, we first create:

- A 2D **GridArchive** where each dimension contains 20 bins across the range
  [-1, 1].
- A **ImprovementEmitter**, which starts from the search point **0** in 10
  dimensional space and a starting Gaussian sampling distribution with standard
  deviation 0.1.
- An **Optimizer** that combines the archive and emitter together.

After initializing the components, we optimize (pyribs maximizes) the negative
10-D Sphere function for 1000 iterations. Users of
[pycma](https://pypi.org/project/cma/) will be familiar with the ask-tell
interface (which pyribs adopted). First, the user must `ask` for new candidate
solutions. After evaluating the solution, they `tell` the algorithm the
objective value and behavior characteristics (BCs) of each candidate solution.
The algorithm then populates the archive and makes decisions on where to sample
solutions from next.

```python
import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer

archive = GridArchive([20, 20], [(-1, 1), (-1, 1)])
emitters = [ImprovementEmitter(archive, [0.0] * 10, 0.1)]
optimizer = Optimizer(archive, emitters)

for itr in range(1000):
    solutions = optimizer.ask()

    objectives = -np.sum(np.square(solutions), axis=1)
    bcs = solutions[:, :2]

    optimizer.tell(objectives, bcs)
```

To visualize this archive with matplotlib, we then use the
`grid_archive_heatmap` function from `ribs.visualize`.

```python
import matplotlib.pyplot as plt
from ribs.visualize import grid_archive_heatmap

grid_archive_heatmap(archive)
plt.show()
```

![Sphere heatmap](readme_assets/sphere_heatmap.png)

For more information, refer to the [documentation](https://docs.pyribs.org/).

## Installation

The pyribs library supports Python 3.6 and greater. Earlier Python versions may
work but are not officially supported.

To install from PyPI, run

```bash
pip install ribs
```

This command only installs dependencies for the core of pyribs. To install
support tools like `ribs.visualize`, run

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

See here for the documentation: <https://docs.pyribs.org>

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

This project is developed and maintained by the
[ICAROS Lab](http://icaros.usc.edu) at USC.

- [Bryon Tjanaka](https://btjanaka.net)
- [Matt Fontaine](https://github.com/tehqin)
- [Yulun Zhang](https://github.com/lunjohnzhang)
- [Sam Sommerer](https://github.com/sam-som-usc)
- Nikitas Klapsis
- [Stefanos Nikolaidis](https://stefanosnikolaidis.net)

## License

The pyribs library is released under the
[MIT License](https://github.com/icaros-usc/pyribs/blob/master/LICENSE).

## Credits

This package was initially created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
