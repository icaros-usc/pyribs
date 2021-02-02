# Examples

```{toctree}
:hidden:

examples/sphere
examples/lunar_lander
```

Examples assume some experience with pyribs and provide commented source code
with fewer explanations than tutorials. If running locally, install dependencies
with:

```bash
pip install ribs[examples]
```

| Name                         | Archive                             | Emitter                                    | Optimizer                           |
| ---------------------------- | ----------------------------------- | ------------------------------------------ | ----------------------------------- |
| {doc}`examples/sphere`       | (several)                           | (several)                                  | {class}`~ribs.optimizers.Optimizer` |
| {doc}`examples/lunar_lander` | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.ImprovementEmitter` | {class}`~ribs.optimizers.Optimizer` |
