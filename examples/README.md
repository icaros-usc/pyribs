# Examples

This directory contains examples for using pyribs. For the most part, these are
Python files with comments on usage and some explanations. For more detailed
explanations of pyribs, the `tutorials` directory contains Jupyter notebooks
with full explanations of pyribs; these are also included in the documentation
at <https://ribs.readthedocs.io/tutorials.html>.

To run these examples, you will need to install some additional dependencies
with `pip install ribs[examples]` or `pip install -e .[examples]`.

## Manifest

| File                  | Description                                                                               |
| --------------------- | ----------------------------------------------------------------------------------------- |
| `run_sphere.py`       | The Sphere function with vanilla MAP-Elites.                                              |
| `run_sphere_cvt.py`   | The Sphere function with CVT-MAP-Elites.                                                  |
| `lunar_lander_cvt.py` | Lunar Lander with CVT-MAP-Elites, using trajectories as BCs and Dask for parallelization. |
