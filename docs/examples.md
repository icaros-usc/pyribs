# Examples

```{toctree}
:hidden:

examples/run_sphere
```

Examples assume some experience with ribs and provide commented source code with
fewer explanations than tutorials. If running locally, install dependencies
with:

```bash
pip install ribs[examples]
```

Below is the list of examples.

| Name                       | Description                          | Archive                             | Emitter                                 | Optimizer                           | BCs |
| -------------------------- | ------------------------------------ | ----------------------------------- | --------------------------------------- | ----------------------------------- | --- |
| {doc}`examples/run_sphere` | The Sphere function with MAP-Elites. | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.GaussianEmitter` | {class}`~ribs.optimizers.Optimizer` |     |
