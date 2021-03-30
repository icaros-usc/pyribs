# pyribs

|             Website              |                     Source                     |                                                       PyPI                                                        |                                                               Conda                                                                |                                                                                                      CI/CD                                                                                                       |                    Docs                    |                                                                   Docs Status                                                                    |                                                                   Twitter                                                                    |
| :------------------------------: | :--------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: |
| [pyribs.org](https://pyribs.org) | [GitHub](https://github.com/icaros-usc/pyribs) | [![PyPI](https://img.shields.io/pypi/v/ribs.svg?style=flat-square&color=blue)](https://pypi.python.org/pypi/ribs) | [![Conda Recipe](https://img.shields.io/badge/recipe-pyribs-green.svg?style=flat-square)](https://anaconda.org/conda-forge/pyribs) | [![Tests](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Ficaros-usc%2Fpyribs%2Fbadge&style=flat-square)](https://github.com/icaros-usc/pyribs/actions?query=workflow%3A"Tests") | [docs.pyribs.org](https://docs.pyribs.org) | [![Documentation Status](https://readthedocs.org/projects/ribs/badge/?version=stable&style=flat-square)](https://readthedocs.org/projects/ribs/) | [![Twitter](https://img.shields.io/badge/twitter-%231DA1F2.svg?&style=flat-square&logo=twitter&logoColor=white)](https://twitter.com/pyribs) |

A _bare-bones_ Python library for quality diversity optimization. pyribs is the
official implementation of the Covariance Matrix Adaptation MAP-Elites (CMA-ME)
algorithm and implements the _Rapid Illumination of Behavior Space (RIBS)_
redesign of MAP-Elites detailed in the paper
[Covariance Matrix Adapation for the Rapid Illumination of Behavior Space](https://arxiv.org/abs/1912.02400).

## Overview

![Types of Optimization](readme_assets/optimization_types.png)

[Quality diversity (QD) optimization](https://arxiv.org/abs/2012.04322) is a
subfield of optimization where solutions generated cover every point in a
behavior space while simultaneously maximizing (or minimizing) a single
objective. QD algorithms within the MAP-Elites family of QD algorithms produce
heatmaps (archives) as output where each cell contains the best discovered
representative of a region in behavior space.

While many QD libraries exist, this particular library aims to be the QD analog
to the [pycma](https://pypi.org/project/cma/) library (a single objective
optimization library). In contrast to other QD libraries, this library is
"bare-bones," meaning pyribs (like [pycma](https://pypi.org/project/cma/))
focuses solely on optimizing fixed-dimensional continuous domains. Focusing
solely on this one commonly-occurring problem allows us to optimize the library
for performance as well as simplicity of use. For applications of QD on discrete
domains, we recommend using [qdpy](https://gitlab.com/leo.cazenille/qdpy/) or
[sferes](https://github.com/sferes2/sferes2).

A user of pyribs selects three components that meet the needs of their
application:

- An **Archive** saves the best representatives generated within behavior space.
- **Emitters** control how new candidate solutions are generated and affect if
  the algorithm prioritizes quality or diversity.
- An **Optimizer** joins the **Archive** and **Emitters** together and acts as a
  scheduling algorithm for emitters. The **Optimizer** provides an interface for
  requesting new candidate solutions and telling the algorithm how candidates
  performed.

## Citation

If you use pyribs in your research, please cite it as follows. Note that you
will need to include the
[hyperref](https://www.overleaf.com/learn/latex/Hyperlinks#Linking_web_addresses)
package in order to use the `\url` command.

```
@misc{pyribs,
  title = {pyribs: A bare-bones Python library for quality diversity
           optimization},
  author = {Bryon Tjanaka and Matthew C. Fontaine and Yulun Zhang and
            Sam Sommerer and Nathan Dennler and Stefanos Nikolaidis},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/icaros-usc/pyribs}},
}
```

If you use the CMA-ME algorithm, please also cite
[Fontaine 2020](https://dl.acm.org/doi/10.1145/3377930.3390232).

```
@inproceedings{10.1145/3377930.3390232,
  author = {Fontaine, Matthew C. and Togelius, Julian and Nikolaidis, Stefanos and Hoover, Amy K.},
  title = {Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space},
  year = {2020},
  isbn = {9781450371285},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3377930.3390232},
  doi = {10.1145/3377930.3390232},
  booktitle = {Proceedings of the 2020 Genetic and Evolutionary Computation Conference},
  pages = {94–102},
  numpages = {9},
  location = {Canc\'{u}n, Mexico},
  series = {GECCO '20}
}
```

## Usage

Here we show an example application of CMA-ME in pyribs. To initialize the
algorithm, we first create:

- A 2D **GridArchive** where each dimension contains 20 bins across the range
  [-1, 1].
- An **ImprovementEmitter**, which starts from the search point **0** in 10
  dimensional space and a Gaussian sampling distribution with standard deviation
  0.1.
- An **Optimizer** that combines the archive and emitter together.

After initializing the components, we optimize (pyribs maximizes) the negative
10-D Sphere function for 1000 iterations. Users of
[pycma](https://pypi.org/project/cma/) will be familiar with the ask-tell
interface (which pyribs adopted). First, the user must `ask` the optimizer for
new candidate solutions. After evaluating the solution, they `tell` the
optimizer the objective value and behavior characteristics (BCs) of each
candidate solution. The algorithm then populates the archive and makes decisions
on where to sample solutions next. Our toy example uses the first two parameters
of the search space as BCs.

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

pyribs supports Python 3.6-3.9. Earlier Python versions may work but are not
officially supported.

To install from PyPI, run

```bash
pip install ribs
```

This command only installs dependencies for the core of pyribs. To install
support tools like `ribs.visualize`, run

```bash
pip install ribs[all]
```

Equivalently, you can install the base version (equivalent to `ribs`) from Conda
with

```bash
conda install -c conda-forge pyribs-base
```

The full version (equivalent to `ribs[all]`) can be installed with

```bash
conda install -c conda-forge pyribs
```

To test your installation, import it and print the version with:

```bash
python -c "import ribs; print(ribs.__version__)"
```

You should see a version number like `0.2.0` in the output.

### From Source

To install a version from source, clone the repo

```bash
git clone https://github.com/icaros-usc/pyribs
```

Then `cd` into it

```bash
cd pyribs
```

And run

```bash
pip install -e .[all]
```

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

pyribs is developed and maintained by the [ICAROS Lab](http://icaros.usc.edu) at
USC.

- [Bryon Tjanaka](https://btjanaka.net)
- [Matt Fontaine](https://github.com/tehqin)
- [Yulun Zhang](https://github.com/lunjohnzhang)
- [Sam Sommerer](https://github.com/sam-sommerer)
- [Nathan Dennler](https://ndennler.github.io/)
- Nikitas Klapsis
- [Stefanos Nikolaidis](https://stefanosnikolaidis.net)

We thank [Amy K. Hoover](http://amykhoover.com/) and
[Julian Togelius](http://julian.togelius.com/) for their contributions deriving
the CMA-ME algorithm.

## License

pyribs is released under the
[MIT License](https://github.com/icaros-usc/pyribs/blob/master/LICENSE).

## Credits

The pyribs package was initially created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
