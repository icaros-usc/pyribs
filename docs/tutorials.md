# Tutorials

```{toctree}
:hidden:

tutorials/lunar_lander
tutorials/tom_cruise_dqd
tutorials/cma_mae
tutorials/lsi_mnist
tutorials/arm_repertoire
tutorials/fooling_mnist
```

Tutorials are Python notebooks with detailed explanations of pyribs usage. They
may be run locally or on
[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). If
running locally, make sure to have
[Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
installed. Each notebook contains cell magic that installs the dependencies it
needs for execution.

## Key Algorithms

These tutorials focus on the key algorithms in pyribs: CMA-ME
([Fontaine 2020](https://arxiv.org/abs/1912.02400)), CMA-MEGA
([Fontaine 2021](https://proceedings.neurips.cc/paper/2021/hash/532923f11ac97d3e7cb0130315b067dc-Abstract.html)),
and CMA-MAE / CMA-MAEGA ([Fontaine 2022](https://arxiv.org/abs/2205.10752)). In
general, we highly recommend CMA-MAE and CMA-MAEGA, and going through these
tutorials will help you understand how these two algorithms work.

| Name                            | Archive                             | Emitter                                             | Scheduler                           |
| ------------------------------- | ----------------------------------- | --------------------------------------------------- | ----------------------------------- |
| {doc}`tutorials/lunar_lander`   | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.EvolutionStrategyEmitter`    | {class}`~ribs.schedulers.Scheduler` |
| {doc}`tutorials/tom_cruise_dqd` | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.GradientArborescenceEmitter` | {class}`~ribs.schedulers.Scheduler` |
| {doc}`tutorials/cma_mae`        | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.EvolutionStrategyEmitter`    | {class}`~ribs.schedulers.Scheduler` |

## Applications

These tutorials focus on applications of pyribs.

| Name                            | Archive                             | Emitter                                          | Scheduler                           |
| ------------------------------- | ----------------------------------- | ------------------------------------------------ | ----------------------------------- |
| {doc}`tutorials/lsi_mnist`      | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.EvolutionStrategyEmitter` | {class}`~ribs.schedulers.Scheduler` |
| {doc}`tutorials/arm_repertoire` | {class}`~ribs.archives.CVTArchive`  | {class}`~ribs.emitters.EvolutionStrategyEmitter` | {class}`~ribs.schedulers.Scheduler` |
| {doc}`tutorials/fooling_mnist`  | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.GaussianEmitter`          | {class}`~ribs.schedulers.Scheduler` |
