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

| Name                         | Archive                             | Emitter                                          | Scheduler                           |
| ---------------------------- | ----------------------------------- | ------------------------------------------------ | ----------------------------------- |
| {doc}`examples/sphere`       | (several)                           | (several)                                        | {class}`~ribs.schedulers.Scheduler` |
| {doc}`examples/lunar_lander` | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.EvolutionStrategyEmitter` | {class}`~ribs.schedulers.Scheduler` |
