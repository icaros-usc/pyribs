# Tutorials

```{toctree}
:hidden:

tutorials/lunar_lander
tutorials/arm_repertoire
tutorials/fooling_mnist
tutorials/training_mnist
```

Tutorials contain detailed explanations of ribs usage in a Python notebook.
These may be run locally with Jupyter, as well as on Colab. If running locally,
make sure to have
[Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
installed. Each notebook contains cell magic that installs the dependencies it
needs for execution.

Below is the list of tutorials, sorted _roughly_ in order of difficulty.

| Name                            | Archive                             | Emitter                                    | Optimizer                           |
| ------------------------------- | ----------------------------------- | ------------------------------------------ | ----------------------------------- |
| {doc}`tutorials/lunar_lander`   | {class}`~ribs.archives.GridArchive` |                                            | {class}`~ribs.optimizers.Optimizer` |
| {doc}`tutorials/arm_repertoire` | {class}`~ribs.archives.CVTArchive`  | {class}`~ribs.emitters.ImprovementEmitter` | {class}`~ribs.optimizers.Optimizer` |
| {doc}`tutorials/fooling_mnist`  | {class}`~ribs.archives.GridArchive` |                                            | {class}`~ribs.optimizers.Optimizer` |
