# Examples

This directory contains examples for using pyribs. The `tutorials` directory
contains Jupyter notebooks with detailed explanations; these are also included
in the documentation on the
[Tutorials](https://docs.pyribs.org/en/stable/tutorials.html) page. To run these
locally, make sure to have
[Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
installed. Each notebook contains cell magic that installs the dependencies it
needs for execution.

The rest of this directory contains Python files with comments on usage and some
explanations. These are intended for slightly more experienced users. For
descriptions of these examples, see the
[Examples](https://docs.pyribs.org/en/stable/examples.html) page in the
documentation. To run these other examples locally, install some additional
dependencies with `pip install ribs[examples]` or `pip install -e .[examples]`.

Finally, `tools` contains tools for making the examples, such as a script for
training an MNIST classifier.
