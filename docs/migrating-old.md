# Migrating to v0.5.0

The new version of pyribs is here! A lot has changed since our last version,
v0.4.0. We've added new features and algorithms, expanded our documentation, and
fixed a number of bugs. These improvements introduced many breaking changes in
our API. Thus, in this guide, we'll review what's new and how to migrate to
v0.5.0 from v0.4.0. For the full list of changes, refer to our
[History page](./history).

## Tutorials

We've added new tutorials and expanded our old ones! We recommend going over the
updated version of our introductory tutorial, the
[Lunar Lander tutorial](./tutorials/lunar_lander), to get a feel for the new
API. Then, check out how we have integrated the
[CMA-MAE](https://arxiv.org/abs/2205.10752) algorithm in our
[CMA-MAE tutorial](./tutorials/cma_mae), and learn how to implement
[DQD algorithms](https://arxiv.org/abs/2106.03894) in our
[Tom Cruise images](./tutorials/tom_cruise_dqd) tutorial.

## Installation

Previously, there were two distributions of pyribs:

- The default distribution, known as `ribs` on PyPI and `pyribs-base` on Conda.
- `all`, known as `ribs[all]` on PyPI `pyribs` on Conda, which included
  Matplotlib.

Now, there are three distributions:

- The default distribution, known as `ribs` on PyPI and `pyribs` on Conda.
- `visualize`, known as `ribs[visualize]` on PyPI and `pyribs-visualize` on
  Conda. This distribution includes visualization dependencies, currently just
  Matplotlib.
- `all`, known as `ribs[all]` on PyPI and `pyribs-all` on Conda. This
  distribution includes dependencies for the `visualize` extra and will include
  any dependencies for additional extras which we add in the future.

We recommend installing the default distribution if you do not need any plotting
utilities, e.g., `pip install ribs` or `conda install pyribs`. If you are using
plotting utilities, you can use `pip install ribs[visualize]` or
`conda install pyribs-visualize`.

## Terminology

- We have adopted the `measures` terminology of recent literature over the
  `behaviors` terminology in pyribs 0.4.0. Names such as `behavior_values` are
  now referred to as `measures`.
- Many of our methods now operate in batch. The batch arguments are referred to
  as `solution_batch`, `objective_batch`, `measures_batch`, and
  `metadata_batch`, while individual arguments are referred to as `solution`,
  `objective`, `measures`, and `metadata`.
- For consistency with the literature, we have replaced the term `bins` with
  `cells` in archives.

## Deprecation of Positional Arguments

Following
[scikit-learn](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep009/proposal.html),
almost all constructor arguments must now be passed in as keyword arguments.
Given the many parameters that these objects have, this makes it easier to see
what each parameter means.

To illustrate, the signature for {class}`~ribs.emitters.EvolutionStrategyEmitter` is:

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

## DQD Algorithms

To support differentiable quality diversity (DQD) algorithms, we have...

## ME-MAP-Elites and BanditScheduler

We have added a new {class}`~ribs.schedulers.BanditScheduler` which maintains a
pool of emitters and only asks a subset of emitters for solutions on each
iteration. The emitters to ask are selected with a multi-armed bandit algorithm.
This scheduler is adapted from the
[ME-MAP-Elites](https://arxiv.org/abs/2007.05352) algorithm.

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

## Miscellaneous

- **Input validation:** Many of our methods, e.g., the archives'
  {meth}`~ribs.archives.ArchiveBase.add` method and schedulers'
  {meth}`~ribs.schedulers.Scheduler.ask` and
  {meth}`~ribs.schedulers.Scheduler.tell` methods, now have input validation to
  help catch common argument errors. For example, we check that arguments are
  the correct shape, and we check that they do not have `NaN` or `inf` values.
- **Deprecation of ribs.factory:** We have now removed `ribs.factory` as we
  believe it is out of scope for the current library.

---

- **Backwards-incompatible:** Implement Scalable CMA-ES Optimizers (#274, #288)
- Make ribs.emitters.opt public (#281)
- Add normalized QD score to ArchiveStats (#276)
- **Backwards-incompatible:** Make ArchiveStats a dataclass (#275)
- Add method for computing CQD score in archives (#252)
- **Backwards-incompatible:** Allow custom initialization in Gaussian and
  IsoLine emitters (#259, #265)
- Implement CMA-MAE archive thresholds (#256, #260)
  - Revive the old implementation of `add_single` removed in (#221)
  - Add separate tests for `add_single` and `add` with single solution
- Add restart timer to `EvolutionStrategyEmitter` and
  `GradientArborescenceEmitter`(#255)
- Rename fields and update documentation (#249, #250)
  - **Backwards-incompatible:** rename `Optimizer` to `Scheduler`
  - **Backwards-incompatible:** rename `objective_value` to `objective`
  - **Backwards-incompatible:** rename `behavior_value`/`bcs` to `measures`
  - **Backwards-incompatible:** `behavior_dim` in archives is now `measure_dim`
  - Rename `n_solutions` to `batch_size` in `Scheduler`.
- Add `GradientArborescenceEmitter`, which is used to implement CMA-MEGA (#240,
  #263, #264, #282, #321)
- Expose `emitter.restarts` as a property (#248)
- Rename methods in ArchiveDataFrame and rename as_pandas behavior columns
  (#236)
- Allow adding solutions one at a time in optimizer (#233)
- `emitter.tell` now takes in `status_batch` and `value_batch` (#227)
- **Backwards-incompatible:** Replaced `ImprovementEmitter`,
  `RandomDirectionEmitter`, and `OptimizingEmitter` with
  `EvolutionStrategyEmitter` (#220, #223, #278)
- Introduced the Ranker object, which is responsible for ranking the solutions
  based on different objectives (#209, #222, #245)
- Add index_of_single method for getting index of measures for one solution
  (#214)
- **Backwards-incompatible:** Add `sigma`, `sigma0` options to
  `gaussian_emitter` and `iso_line_emitter` (#199)
  - `gaussian_emitter` constructor requires `sigma`; `sigma0` is optional.
  - `iso_line_emitter` constructor takes in optional parameter `sigma0`.
- **Backwards-incompatible:** Add EliteBatch and rename fields in Elite (#191)
