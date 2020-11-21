# Usage

The ribs usage guide is divided into two parts:

- **Tutorials** contain detailed explanations of ribs usage in a Python
  notebook. These may be run locally with Jupyter, as well as on Colab.
- **Examples** assume more experience with ribs and provide commented source
  code with fewer examples.

If running any of these locally, make sure to install the `examples` extra with:

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
