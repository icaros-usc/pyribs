======
pyribs
======

.. image:: https://img.shields.io/pypi/v/ribs.svg
        :target: https://pypi.python.org/pypi/ribs

.. .. image:: https://img.shields.io/travis/icaros-usc/ribs.svg
..         :target: https://travis-ci.com/icaros-usc/ribs

.. image:: https://readthedocs.org/projects/ribs/badge/?version=latest
        :target: https://ribs.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

A *bare-bones* quality diversity optimization library. The algorithms implemented here enable *Rapid Illumination of Behavior Spaces (RIBS)*.

Documentation: https://ribs.readthedocs.io

Installation
------------

pyribs supports Python 3.6 and greater. Earlier versions may work but are not
officially supported.

To install from PyPI, run ::

  pip install ribs

To install a development version, clone this repo, cd into it, and run ::

  pip install -e .

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

$ make servedocs

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
