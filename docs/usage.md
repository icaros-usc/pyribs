# Usage

pyribs has _tutorials_, which contain detailed explanations of code in a Python
notebook, as well as _examples_, which simply present code with (detailed)
comments. Both can be run locally, but the tutorials may also be viewed on
Colab. If you are running locally, make sure to install the `examples` extra
with:

```bash
pip install ribs[examples]
```

Alternatively, if you have cloned the source code for ribs, you can also run:

```bash
pip install -e .[examples]
```

## Tutorials

| Name                                | Description                                                          | Archive                             | Emitter | Optimizer | BCs |
| ----------------------------------- | -------------------------------------------------------------------- | ----------------------------------- | ------- | --------- | --- |
| {doc}`tutorials/lunar_lander`       | Use CMA-ME to train an agent in the `LunarLander-v2` OpenAI Gym env. | {class}`~ribs.archives.GridArchive` |         |           |     |
| {doc}`tutorials/inverse_kinematics` | Train a robotic arm to move to different positions.                  |                                     |         |           |     |

## Examples

| Name                       | Description                          | Archive                             | Emitter                                 | Optimizer                           | BCs |
| -------------------------- | ------------------------------------ | ----------------------------------- | --------------------------------------- | ----------------------------------- | --- |
| {doc}`examples/run_sphere` | The Sphere function with MAP-Elites. | {class}`~ribs.archives.GridArchive` | {class}`~ribs.emitters.GaussianEmitter` | {class}`~ribs.optimizers.Optimizer` |     |
