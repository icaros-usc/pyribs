======
pyribs
======

+---------+------------------------------------------------------------------------------------------------+
| Source  | `GitHub <https://github.com/icaros-usc/pyribs>`_                                               |
+---------+------------------------------------------------------------------------------------------------+
| PyPI    | .. image:: https://img.shields.io/pypi/v/ribs.svg                                              |
|         |     :target: https://pypi.python.org/pypi/ribs                                                 |
|         |     :alt: PyPI                                                                                 |
+---------+------------------------------------------------------------------------------------------------+
| CI/CD   | .. image:: https://github.com/icaros-usc/pyribs/workflows/Automated%20Testing/badge.svg)]      |
|         |     :target: https://github.com/icaros-usc/pyribs/actions?query=workflow%3A"Automated+Testing" |
|         |     :alt: Automated Testing                                                                    |
+---------+------------------------------------------------------------------------------------------------+
| Docs    | `ribs.readthedocs.io <https://ribs.readthedocs.io>`_                                           |
+---------+------------------------------------------------------------------------------------------------+
| Docs    | .. image:: https://readthedocs.org/projects/ribs/badge/?version=latest                         |
| Status  |     :target: https://readthedocs.org/projects/ribs/                                            |
|         |     :alt: Documentation Status                                                                 |
+---------+------------------------------------------------------------------------------------------------+

*Bare-bones* implementations of Quality Diversity algorithms, i.e. algorithms that provide *Rapid Illumination of Behavior Spaces (RIBS)*.

Documentation: https://ribs.readthedocs.io

Installation
------------

pyribs supports Python 3.6 and greater. Earlier versions may work but are not
officially supported.

To install from PyPI, run ::

  pip install ribs

This command only installs dependencies for the core of ribs. To be able to use
tools like ``ribs.visualize``, run ::

  pip install ribs[all]

To install a development version, clone this repo, cd into it, and run ::

  pip install -e .[all]

To test your installation, run ::

  python -c "import ribs; print(ribs.__version__)"

This will import pyribs and print its current version.

Features
--------

* TODO

Documentation
-------------

See here for the documentation: https://ribs.readthedocs.io

To serve the documentation locally, clone the repo and run::

  make servedocs

This will open a window in your browser with the documentation automatically
loaded. Furthermore, every time you make changes to the documentation, the
preview will also reload.

Contributors
------------

This project was completed in the `ICAROS Lab <http://icaros.usc.edu>`_ at USC.

* `Bryon Tjanaka <https://btjanaka.net>`_
* `Matt Fontaine <https://github.com/tehqin>`_
* `Sam Sommerer <https://github.com/sam-som-usc>`_
* `Stefanos Nikolaidis <https://stefanosnikolaidis.net>`_

License
-------

pyribs is released under the `MIT License <https://github.com/icaros-usc/pyribs/blob/master/LICENSE>`_.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
