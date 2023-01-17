# Tutorials

```{toctree}
:hidden:

tutorials/lunar_lander
tutorials/cma_mae
tutorials/tom_cruise_dqd
tutorials/lsi_mnist
tutorials/arm_repertoire
tutorials/fooling_mnist
```

Tutorials are Python notebooks with detailed explanations of pyribs usage. They
may be [run locally](running-locally) or on
[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). Each
tutorial page has a link to open the tutorial in Google Colab.

## Key Algorithms

We recommend new users start with these tutorials which demonstrate how to use
the key algorithms in pyribs.

- {doc}`tutorials/lunar_lander`: Covers the CMA-ME algorithm and various basic
  library features.
- {doc}`tutorials/cma_mae`: Shows how to implement CMA-MAE, a powerful
  algorithm built on CMA-ME, on the sphere linear projection benchmark.
- {doc}`tutorials/tom_cruise_dqd`: Covers CMA-MEGA and CMA-MAEGA, two algorithms
  designed for differentiable quality diversity problems (QD problems where
  gradients are available).

## Applications

The following tutorials show how pyribs can implement a variety of algorithms
for different problems.

- {doc}`tutorials/lsi_mnist`: Generates diverse handwritten MNIST digits with
  CMA-ME.
- {doc}`tutorials/arm_repertoire`: Combines CMA-ME with a CVTArchive to search
  for optimal configurations for a robot arm.
- {doc}`tutorials/fooling_mnist`: Searches for misclassified MNIST images with
  MAP-Elites.

<!-- How MyST handles section labels: https://jupyterbook.org/en/stable/content/references.html -->

(running-locally)=

## Running Locally

If you would like to run the tutorials locally, follow these instructions:

1. Download the notebooks from GitHub
   [here](https://github.com/icaros-usc/pyribs/tree/master/examples/tutorials).
2. Install
   [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html):
   ```bash
   pip install jupyterlab
   ```
3. Start Jupyter Lab. This should open a window in your browser.
   ```bash
   jupyter lab
   ```
4. Open the notebook from within the Jupyter Lab browser window.

Note that each notebook contains cell magic that installs the dependencies it
needs for execution, so even if you have not installed the dependencies on your
own, running the notebook will install the dependencies for you.
