# Tutorials

```{toctree}
:hidden:

tutorials/lunar_lander
tutorials/arm_repertoire
tutorials/fooling_mnist
```

Tutorials are Python notebooks with detailed explanations of pyribs usage. They
may be run locally or on Colab. If running locally, make sure to have
[Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
installed. Each notebook contains cell magic that installs the dependencies it
needs for execution.

| Name                            | Archive                             | Emitter                                    | Optimizer                           |
| ------------------------------- | ----------------------------------- | ------------------------------------------ | ----------------------------------- |
| {doc}`tutorials/lunar_lander`   | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.ImprovementEmitter` | {class}`~ribs.optimizers.Optimizer` |
| {doc}`tutorials/arm_repertoire` | {class}`~ribs.archives.CVTArchive`  | {class}`~ribs.emitters.ImprovementEmitter` | {class}`~ribs.optimizers.Optimizer` |
| {doc}`tutorials/fooling_mnist`  | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.GaussianEmitter`    | {class}`~ribs.optimizers.Optimizer` |
