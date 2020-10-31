# pyribs

|                     Source                     |                                         PyPI                                         |                                                                                          CI/CD                                                                                          |                        Docs                        |                                                          Docs Status                                                           |
| :--------------------------------------------: | :----------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
| [GitHub](https://github.com/icaros-usc/pyribs) | [![pypi](https://img.shields.io/pypi/v/ribs.svg)](https://pypi.python.org/pypi/ribs) | [![Automated Testing](https://github.com/icaros-usc/pyribs/workflows/Automated%20Testing/badge.svg)](https://github.com/icaros-usc/pyribs/actions?query=workflow%3A"Automated+Testing") | [ribs.readthedocs.io](https://ribs.readthedocs.io) | [![Documentation Status](https://readthedocs.org/projects/ribs/badge/?version=latest)](https://readthedocs.org/projects/ribs/) |

_Bare-bones_ implementations of Quality Diversity algorithms, i.e. algorithms
that provide _Rapid Illumination of Behavior Spaces (RIBS)_.

Documentation: https://ribs.readthedocs.io

## Installation

pyribs supports Python 3.6 and greater. Earlier versions may work but are not
officially supported.

To install from PyPI, run:

```bash
pip install ribs
```

To install a development version, clone this repo, cd into it, and run:

```bash
pip install -e .
```

To test your installation, run:

```
python -c "import ribs; print(ribs.__version__)"
```

This will import pyribs and print its current version.

## Features

- TODO

## Documentation

See here for the documentation: https://ribs.readthedocs.io

To serve the documentation locally, clone the repo and run:

```bash
make servedocs
```

This will open a window in your browser with the documentation automatically
loaded. Furthermore, every time you make changes to the documentation, the
preview will also reload.

## Contributors

This project was completed in the `ICAROS Lab <http://icaros.usc.edu>`\_ at USC.

- [Bryon Tjanaka](https://btjanaka.net)
- [Matt Fontaine](https://github.com/tehqin)
- [Sam Sommerer](https://github.com/sam-som-usc)
- [Stefanos Nikolaidis](https://stefanosnikolaidis.net)

## License

pyribs is released under the
[MIT License](https://github.com/icaros-usc/pyribs/blob/master/LICENSE).

## Credits

This package was originally created with
[Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
