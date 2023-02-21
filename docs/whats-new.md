# What's New in v0.5.0

The new version of pyribs is here! A lot has changed since our last version,
v0.4.0. We've added new features and algorithms, expanded our documentation,
changed terminology, and fixed a number of bugs. These improvements introduced
many breaking changes in our API. Thus, in this guide, we'll review what's new
in v0.5.0 compared to v0.4.0. For the full list of changes, refer to our
[History page](./history).

## Tutorials

We've added new tutorials and expanded our old ones! We recommend going over the
updated version of our introductory tutorial, {doc}`tutorials/lunar_lander`, to
get a feel for the new API. Then, check out how we have integrated the
[CMA-MAE](https://arxiv.org/abs/2205.10752) algorithm in
{doc}`tutorials/cma_mae`, and learn how to implement
[Differentiable Quality Diversity (DQD) algorithms](https://arxiv.org/abs/2106.03894)
in {doc}`tutorials/tom_cruise_dqd`.

## Installation

<!-- Previously, there were two distributions of pyribs: -->

<!-- - The default distribution, known as `ribs` on PyPI and `pyribs-base` on Conda. -->
<!-- - `all`, known as `ribs[all]` on PyPI `pyribs` on Conda, which included -->
<!--   Matplotlib. -->

There are now three distributions of pyribs:

| Name      | PyPI Package      | Conda Package      | Description                                                                             |
| --------- | ----------------- | ------------------ | --------------------------------------------------------------------------------------- |
| Default   | `ribs`            | `pyribs`           | Basic pyribs package.                                                                   |
| Visualize | `ribs[visualize]` | `pyribs-visualize` | Adds visualization dependencies, currently just Matplotlib.                             |
| All       | `ribs[all]`       | `pyribs-all`       | Installs dependencies for all extra distributions (currently just the Visualize extra). |

We recommend installing the default distribution if you do not need any plotting
utilities, e.g., `pip install ribs` or `conda install pyribs`. If you are using
plotting utilities, you can use `pip install ribs[visualize]` or
`conda install pyribs-visualize`.

## Terminology

- **measures vs behaviors:** We have adopted the `measures` terminology of
  recent literature over the `behaviors` terminology in pyribs 0.4.0. Names such
  as `behavior_values` are now referred to as `measures`.
- **Batch arguments:** Many of our methods now operate in batch. The batch
  arguments are referred to as `solution_batch`, `objective_batch`,
  `measures_batch`, and `metadata_batch`, while individual arguments are
  referred to as `solution`, `objective`, `measures`, and `metadata`.
- **cells vs bins:** For consistency with the literature, we have replaced the
  term `bins` with `cells` in archives.


## New Emitters

### EvolutionStrategyEmitter

The `ImprovementEmitter`, `RandomDirectionEmitter`, and `OptimizingEmitter` have
been replaced with the {class}`~ribs.emitters.EvolutionStrategyEmitter` which
can replicate any of these earlier emitters' behaviors by selecting an
appropriate _ranker_ from the {mod}`ribs.emitters.rankers` module. For example:

```python
EvolutionStrategyEmitter(archive, ..., ranker="2imp")  # Equivalent to ImprovementEmitter.
EvolutionStrategyEmitter(archive, ..., ranker="imp")  # Single-stage improvement ranking as is used in CMA-MAE.
EvolutionStrategyEmitter(archive, ..., ranker="2rd")  # Two-stage random direction ranking.
EvolutionStrategyEmitter(archive, ..., ranker="obj")  # Objective ranking as was done in OptimizingEmitter.
```

`EvolutionStrategyEmitter` also supports evolution strategies other than CMA-ES
as described in [Tjanaka 2022](https://arxiv.org/abs/2210.02622). These
evolution strategies are available in the {mod}`ribs.emitters.opt` module. For
example:

```python
EvolutionStrategyEmitter(archive, ..., es="sep_cma_es")  # sep-CMA-ES instead of CMA-ES.
```

### GradientArborescenceEmitter

Furthermore, we have added a {class}`~ribs.emitters.GradientArborescenceEmitter`
which supports [DQD algorithms](https://arxiv.org/abs/2106.03894) as is reviewed
in {doc}`tutorials/tom_cruise_dqd`.

## Custom Initial Solutions in Emitters

{class}`~ribs.emitters.GaussianEmitter` and
{class}`~ribs.emitters.IsoLineEmitter` now support passing in custom initial
solutions. By default, they sample from Gaussian distributions initially, but
this can be restricting given that many MAP-Elites variants begin by sampling
from a uniform distribution. With this change, it is now possible to sample
solutions in advance and pass them to the emitter.

```python
# Initially (i.e., when the archive is empty), the emitter will return
# `initial_solutions` during the ask() method.
GaussianEmitter(archive, ..., initial_solutions=[[0.0, 1.0], [1.3, 2.0]])
```

## ME-MAP-Elites and BanditScheduler

We have added a new {class}`~ribs.schedulers.BanditScheduler` which maintains a
pool of emitters and only asks a subset of emitters for solutions on each
iteration. The emitters to ask are selected with a multi-armed bandit algorithm.
This scheduler is adapted from the
[ME-MAP-Elites](https://arxiv.org/abs/2007.05352) algorithm.

## Deprecation of Positional Arguments

Following
[scikit-learn](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep009/proposal.html),
almost all constructor arguments must now be passed in as keyword arguments.
Given the many parameters that these objects have, this makes it easier to see
what each parameter means.

To illustrate, the signature for
{class}`~ribs.emitters.EvolutionStrategyEmitter` is:

```python
EvolutionStrategyEmitter(archive, *, x0, sigma0, ranker='2imp', es='cma_es', es_kwargs=None, selection_rule='filter', restart_rule='no_improvement', bounds=None, batch_size=None, seed=None)
```

All parameters after the `*` are keyword-only. The following will result in an
error:

```python
EvolutionStrategyEmitter(archive, np.zeros(10), 0.1)
```

While the following will be accepted:

```python
EvolutionStrategyEmitter(archive, x0=np.zeros(10), sigma=0.1)
```

## Removal of initialize() Method for Archives

Previously, archives required calling `initialize(solution_dim)` after being
constructed -- this was something that the Optimizer/Scheduler would do. Now,
archives are directly constructed with the `solution_dim` for simplicity.

```python
# v0.4.0 (OLD)
archive = GridArchive(...)
archive.initialize(solution_dim)  # Would be called by Optimizer / Scheduler.

# v0.5.0 (NEW)
archive = GridArchive(solution_dim, ...)
```

## Batched Archive Operations

Before, pyribs archives operated on solutions one at a time. Now, most archives
(with the exception of SlidingBoundariesArchive) rely on batch operations. The
following table shows the old methods and the names of the new methods which
operate on batched inputs. As a convenience, we have also included "single"
methods which can operate on single inputs.

| Old Method            | New Batched Method                               | New Single Method                                  |
| --------------------- | ------------------------------------------------ | -------------------------------------------------- |
| `add()`               | {meth}`~ribs.archives.ArchiveBase.add`           | {meth}`~ribs.archives.ArchiveBase.add_single`      |
| `get_index()`         | {meth}`~ribs.archives.ArchiveBase.index_of`      | {meth}`~ribs.archives.ArchiveBase.index_of_single` |
| `get_random_elite()`  | {meth}`~ribs.archives.ArchiveBase.sample_elites` | N/A                                                |
| `elite_with_behavior` | {meth}`~ribs.archives.ArchiveBase.retrieve`      | {meth}`~ribs.archives.ArchiveBase.retrieve_single` |

To elaborate on these changes:

- All archive indices must now be integers (before, indices could be tuples of
  integers). Thus, the `get_index()` method has been replaced with
  {meth}`~ribs.archives.ArchiveBase.index_of`, which takes in a batch of
  measures (i.e. a `(batch_size, measure_dim)` array) and returns a batch of
  integer indices. Furthermore, whereas {class}`~ribs.archives.ArchiveBase` used
  to take in a `storage_dims` argument, it now takes in a single `storage_dim`
  argument because indices are all integers.
  - Since grid indices have meaning in {class}`~ribs.archives.GridArchive` and
    {class}`~ribs.archives.SlidingBoundariesArchive`, we have also added a
    {meth}`~ribs.archives.GridArchive.grid_to_int_index` and
    {meth}`~ribs.archives.GridArchive.int_to_grid_index` method to convert to
    and from grid indices in these archives.
- {meth}`~ribs.archives.ArchiveBase.add` used to operate on single solutions.
  Now, it inserts solutions into the archive in batch. There is also an
  {meth}`ribs.archives.ArchiveBase.add_single` which inserts solutions one at a
  time and is more amenable to modifications than the batched `add()` method.
- `get_random_elite()`, which sampled one elite from the archive, has been
  replaced with a batched {meth}`~ribs.archives.ArchiveBase.sample_elites`
  method which samples multiple elites at once.
- The `elite_with_behavior()` method which retrieved one elite with given
  measures has been replaced with {meth}`~ribs.archives.ArchiveBase.retrieve`,
  which retrieves a batch of such elites.

Some of these methods also include "single" versions which operate on single
solutions, e.g., {meth}`~ribs.archives.ArchiveBase.index_of_single` returns the
integer index of a single measures array.

## More Flexible Visualization Tools

In all heatmap visualization tools, we have made the colorbar more flexible by
adding a `cbar` option to control which axes the colorbar appears on and a
`cbar_kwargs` option to pass arguments directly to Matplotlib's
{meth}`~matplotlib.pyplot.colorbar`.

```python
from ribs.visualize import grid_archive_heatmap  # cvt_archive_heatmap and sliding_boundaries_archive_heatmap also work

archive = ...

grid_archive_heatmap(archive, cbar="auto")  # Display the colorbar as part of the current axes (default).
grid_archive_heatmap(archive, cbar=None)  # Don't display the colorbar at all.
grid_archive_heatmap(archive, cbar=cbar_ax)  # Display colorbar on a custom Axes.
grid_archive_heatmap(archive, ..., cbar_kwargs={...})  # Pass arguments to the colorbar.
```

In addition:

- We have added an `aspect` argument which can set the aspect ratio of the
  heatmap, i.e. the ratio `height / width`.
- We now support heatmaps for 1D grid archives.

## Continuous Quality Diversity (CQD) Score

We support computing the
[Continuous Quality Diversity (CQD) Score](https://dl.acm.org/doi/10.1145/3520304.3534018)
in all archives with the {meth}`~ribs.archives.ArchiveBase.cqd_score` method.

## Optimizers are now Schedulers

To better reflect their role, the "optimizers" from pyribs v0.4.0 are now
referred to as "schedulers". All schedulers are under the {mod}`ribs.schedulers`
module, including {class}`~ribs.schedulers.Scheduler` and
{class}`~ribs.schedulers.BanditScheduler`.

## ArchiveDataFrame Field Renaming

To reflect the terminology changes and the move to integer indices in archives,
columns in {class}`~ribs.archives.ArchiveDataFrame` (returned by
{meth}`~ribs.archives.ArchiveBase.as_pandas`) are as follows:

Before, they were:

We have renamed our methods as follows:

## Input Validation

**Input validation:** Many of our methods, e.g., the archives'
{meth}`~ribs.archives.ArchiveBase.add` method and schedulers'
{meth}`~ribs.schedulers.Scheduler.ask` and
{meth}`~ribs.schedulers.Scheduler.tell` methods, now have input validation to
help catch common argument errors. For example, we check that arguments are the
correct shape, and we check that they do not have `NaN` or `inf` values.

## Deprecation of ribs.factory

We have now removed `ribs.factory` as we
  believe it is out of scope for the current library.

## ArchiveStats

The {class}`~ribs.archives.ArchiveStats` object now includes
  the normalized QD score (i.e., the QD score divided by the number of elites in
  the archive). Furthermore, it is now a dataclass rather than a namedtuple.


## EliteBatch

We have added an {class}`~ribs.archives.EliteBatch` class to
  represent batches of elites.