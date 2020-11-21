# pyribs

|                     Source                     |                                                  PyPI                                                  |                                                                                                                  CI/CD                                                                                                                   |                        Docs                        |                                                                   Docs Status                                                                    |
| :--------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: |
| [GitHub](https://github.com/icaros-usc/pyribs) | [![PyPI](https://img.shields.io/pypi/v/ribs.svg?style=flat-square)](https://pypi.python.org/pypi/ribs) | [![Automated Testing](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Ficaros-usc%2Fpyribs%2Fbadge&style=flat-square)](https://github.com/icaros-usc/pyribs/actions?query=workflow%3A"Automated+Testing") | [ribs.readthedocs.io](https://ribs.readthedocs.io) | [![Documentation Status](https://readthedocs.org/projects/ribs/badge/?version=latest&style=flat-square)](https://readthedocs.org/projects/ribs/) |

A _bare-bones_ quality diversity optimization library. The algorithms
implemented here enable _Rapid Illumination of Behavior Spaces (RIBS)_.

## Installation

pyribs supports Python 3.6 and greater. Earlier versions may work but are not
officially supported.

To install from PyPI, run

```bash
pip install ribs
```

This command only installs dependencies for the core of ribs. To be able to use
tools like `ribs.visualize`, run

```bash
pip install ribs[all]
```

To install a development version, clone this repo, cd into it, and run

```bash
pip install -e .[all]
```

To test your installation, run

```bash
python -c "import ribs; print(ribs.__version__)"
```

This will import pyribs and print its current version.

## Features

- TODO

## Documentation

See here for the documentation: <https://ribs.readthedocs.io>

To serve the documentation locally, clone the repo and run

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
- [Sam Sommerer](https://github.com/sam-som-usc)
- [Stefanos Nikolaidis](https://stefanosnikolaidis.net)
- [Yulun Zhang](https://github.com/lunjohnzhang)

## License

pyribs is released under the
[MIT License](https://github.com/icaros-usc/pyribs/blob/master/LICENSE).

## Credits

This package was initially created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
