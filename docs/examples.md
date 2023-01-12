# Examples

```{toctree}
:hidden:

examples/sphere
examples/lunar_lander
```

These examples provide single Python files with fewer explanations than
tutorials. If running locally, install dependencies with:

```bash
pip install ribs[examples]
```

Here are the current examples:

- {doc}`examples/sphere`: Demonstrates how to set up recent QD algorithms and
  apply them to the sphere benchmark function.
- {doc}`examples/lunar_lander`: An extended version of the
  [Lunar Lander tutorial](tutorials/lunar_lander) which speeds up evaluations by
  distributing them across multiple CPUs.
