# Tutorials

```{toctree}
:hidden:

tutorials/lunar_lander
tutorials/arm_repertoire
```

Tutorials contain detailed explanations of ribs usage in a Python notebook.
These may be run locally with Jupyter, as well as on Colab. If running locally,
install dependencies with:

```bash
pip install ribs[examples]
```

Below is the list of tutorials, sorted _roughly_ in order of difficulty.

| Name                            | Archive                             | Emitter                                | Optimizer                           |
| ------------------------------- | ----------------------------------- | -------------------------------------- | ----------------------------------- |
| {doc}`tutorials/lunar_lander`   | {class}`~ribs.archives.GridArchive` |                                        | {class}`~ribs.optimizers.Optimizer` |
| {doc}`tutorials/arm_repertoire` | {class}`~ribs.archives.CVTArchive`  | {class}`~ribs.emitters.IsoLineEmitter` | {class}`~ribs.optimizers.Optimizer` |
