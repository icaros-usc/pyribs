# Tutorials

Tutorials are Python notebooks with detailed explanations of pyribs usage. They
may be [run locally](running-locally) or on
[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). Each
tutorial page has a link to open the tutorial in Google Colab.

## Algorithms

```{toctree}
:hidden:

tutorials/lunar_lander
tutorials/cma_mae
tutorials/tom_cruise_dqd
tutorials/qdhf
tutorials/scalable_cma_mae
```

These tutorials demonstrate how to use the key algorithms in pyribs. We
recommend that new users start with the lunar lander tutorial and one or more of
the other algorithm tutorials.

- {doc}`tutorials/lunar_lander`: Our introductory tutorial. Covers the CMA-ME
  algorithm and various basic library features.
- {doc}`tutorials/cma_mae`: Shows how to implement CMA-MAE, a powerful algorithm
  built on CMA-ME, on the sphere linear projection benchmark.
- {doc}`tutorials/tom_cruise_dqd`: Covers CMA-MEGA and CMA-MAEGA, two algorithms
  designed for differentiable quality diversity problems (QD problems where
  gradients are available).
- {doc}`tutorials/qdhf`: Illustrates how to implement the QDHF algorithm on the
  problem of latent space illumination with a stable diffusion model.
- {doc}`tutorials/scalable_cma_mae`: How to use variants of CMA-MAE that scale
  to thousands or even millions of parameters.

## Applications

```{toctree}
:hidden:

tutorials/lsi_mnist
tutorials/arm_repertoire
tutorials/fooling_mnist
tutorials/optuna
```

The following tutorials show how pyribs can be applied to a variety of problems.

- {doc}`tutorials/lsi_mnist`: Generates diverse handwritten MNIST digits with
  CMA-ME.
- {doc}`tutorials/arm_repertoire`: Combines CMA-ME with a CVTArchive to search
  for optimal configurations for a robot arm.
- {doc}`tutorials/fooling_mnist`: Searches for misclassified MNIST images with
  MAP-Elites.
- {doc}`tutorials/optuna`: Details on how pyribs can be integrated with the
  [Optuna](https://optuna.org) framework for hyperparameter optimization.

<!--

## Features

Finally, these tutorials provide a closer look at some of the features of
pyribs.

```{toctree}
:hidden:

tutorials/features/example
```

- {doc}`tutorials/features/example`: Placeholder for upcoming tutorials!

-->

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
