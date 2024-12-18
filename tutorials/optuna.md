# Integrating with Optuna

[Optuna](https://optuna.org/) is an open source hyperparameter optimization
framework. Currently, there is limited support for using pyribs in Optuna via
plugins that are available on [OptunaHub](https://hub.optuna.org):

- [CMA-MAE Sampler](https://hub.optuna.org/samplers/cmamae/) provides an Optuna
  sampler that uses CMA-MAE as implemented in pyribs. This enables one to search
  for diverse, high-performing hyperparameters using CMA-MAE.
- [Pyribs Visualization Wrappers](https://hub.optuna.org/visualization/plot_pyribs/)
  provides wrappers around pyribs visualization tools, enabling visualizing
  results from the CMA-MAE sampler.

For more information, please refer to the documentation for these plugins and to
the more general Optuna documentation.
