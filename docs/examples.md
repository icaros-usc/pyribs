# Examples

```{toctree}
:hidden:

examples/sphere
examples/lunar_lander
examples/bop_elites
examples/cqd_score
```

These examples provide single Python files with fewer explanations than
tutorials. To run each example, first install the dependencies listed at the top
of each file, then follow the usage instructions. Here are the current examples:

- {doc}`examples/sphere`: Demonstrates how to set up recent QD algorithms and
  apply them to the sphere linear projection benchmark.
- {doc}`examples/lunar_lander`: An extended version of the
  [Lunar Lander tutorial](tutorials/lunar_lander) which distributes evaluations
  with Dask and adds other features such as a command-line interface.
- {doc}`examples/bop_elites`: An example of how to run Bayesian Optimization of
  Elites (BOP-Elites), an algorithm that adapts methods from Bayesian
  optimization to model the objective and measure functions and select solutions
  to evaluate.
- {doc}`examples/cqd_score`: An example of how to compute Continuous QD Score, a
  metric that accounts for the tradeoff between objective values and distance to
  desired measure values.
