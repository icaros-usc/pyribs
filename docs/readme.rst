======
pyribs
======

+--------------------------------------------------+---------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------+------------------------------------------------------------------------------------------+
| Source                                           | PyPI                                                                | CI/CD                                                                                                                                    | Docs                                                 | Docs Status                                                                              |
+==================================================+=====================================================================+==========================================================================================================================================+======================================================+==========================================================================================+
| `GitHub <https://github.com/icaros-usc/pyribs>`_ | .. image:: https://img.shields.io/pypi/v/ribs.svg?style=flat-square | .. image:: https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Ficaros-usc%2Fpyribs%2Fbadge&style=flat-square | `ribs.readthedocs.io <https://ribs.readthedocs.io>`_ | .. image:: https://readthedocs.org/projects/ribs/badge/?version=latest&style=flat-square |
|                                                  |     :target: https://pypi.python.org/pypi/ribs                      |     :target: https://github.com/icaros-usc/pyribs/actions?query=workflow%3A"Automated+Testing"                                           |                                                      |     :target: https://readthedocs.org/projects/ribs/                                      |
|                                                  |     :alt: PyPI                                                      |     :alt: Automated Testing                                                                                                              |                                                      |     :alt: Documentation Status                                                           |
+--------------------------------------------------+---------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------+------------------------------------------------------------------------------------------+

*Bare-bones* implementations of Quality Diversity algorithms, i.e. algorithms that provide *Rapid Illumination of Behavior Spaces (RIBS)*.

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
