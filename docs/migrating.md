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

## DQD Algorithms

To support differentiable quality diversity (DQD) algorithms, we have...

## ME-MAP-Elites and BanditScheduler

We have added a new {class}`~ribs.schedulers.BanditScheduler` which maintains a
pool of emitters and only asks a subset of emitters for solutions on each
iteration. The emitters to ask are selected with a multi-armed bandit algorithm.
This scheduler is adapted from the
[ME-MAP-Elites](https://arxiv.org/abs/2007.05352) algorithm.

## Faster Archives

## Miscellaneous Improvements

- **Input validation:** Many of our methods, e.g., the archives'
  {meth}`~ribs.archives.ArchiveBase.add` method and schedulers'
  {meth}`~ribs.schedulers.Scheduler.ask` and
  {meth}`~ribs.schedulers.Scheduler.tell` methods, now have input validation to
  help catch common argument errors. For example, we check that arguments are
  the correct shape, and we check that they do not have `NaN` or `inf` values.

---

- **Backwards-incompatible:** Implement Scalable CMA-ES Optimizers (#274, #288)
- Make ribs.emitters.opt public (#281)
- Add normalized QD score to ArchiveStats (#276)
- **Backwards-incompatible:** Make ArchiveStats a dataclass (#275)
- Add method for computing CQD score in archives (#252)
- **Backwards-incompatible:** Deprecate positional arguments in constructors
  (#261)
- **Backwards-incompatible:** Allow custom initialization in Gaussian and
  IsoLine emitters (#259, #265)
- Implement CMA-MAE archive thresholds (#256, #260)
  - Revive the old implementation of `add_single` removed in (#221)
  - Add separate tests for `add_single` and `add` with single solution
- Fix all examples and tutorials (#253)
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
- Update emitter `tell()` docstrings to no longer say "Inserts entries into
  archive" (#247)
- Expose `emitter.restarts` as a property (#248)
- Specify that `x0` is 1D for all emitters (#244)
- Add `best_elite` property for archives (#237)
- Rename methods in ArchiveDataFrame and rename as_pandas behavior columns
  (#236)
- Re-run CVTArchive benchmarks and update CVTArchive (#235)
  - **Backwards-incompatible:** `use_kd_tree` now defaults to True since the k-D
    tree is always faster than brute force in benchmarks.
- Allow adding solutions one at a time in optimizer (#233)
- Minimize numba usage (#232)
- **Backwards-incompatible:** Implement batch addition in archives (#221, #242)
  - `add` now adds a batch of solutions to the archive
  - `add_single` adds a single solution
- `emitter.tell` now takes in `status_batch` and `value_batch` (#227)
- Make epsilon configurable in archives (#226)
- **Backwards-incompatible:** Remove ribs.factory (#225, #228)
- **Backwards-incompatible:** Replaced `ImprovementEmitter`,
  `RandomDirectionEmitter`, and `OptimizingEmitter` with
  `EvolutionStrategyEmitter` (#220, #223, #278)
- Introduced the Ranker object, which is responsible for ranking the solutions
  based on different objectives (#209, #222, #245)
- Add index_of_single method for getting index of measures for one solution
  (#214)
- **Backwards-incompatible:** Replace elite_with_behavior with retrieve and
  retrieve_single in archives (#213, #215, #295)
- **Backwards-incompatible:** Replace get_index with batched index_of method in
  archives (#208)
  - Also added `grid_to_int_index` and `int_to_grid_index` methods for
    `GridArchive` and `SlidingBoundariesArchive`
- **Backwards-incompatible:** Made it such that each archive is initialized
  fully in its constructor instead of needing a separate
  .initialize(solution_dim) call (#200)
- **Backwards-incompatible:** Add `sigma`, `sigma0` options to
  `gaussian_emitter` and `iso_line_emitter` (#199)
  - `gaussian_emitter` constructor requires `sigma`; `sigma0` is optional.
  - `iso_line_emitter` constructor takes in optional parameter `sigma0`.
- **Backwards-incompatible:** Add `cbar`, `aspect` options for
  `cvt_archive_heatmap` (#197)
- **Backwards-incompatible:** Add `aspect` option to `grid_archive_heatmap` +
  support for 1D heatmaps (#196)
  - `square` option no longer works
- **Backwards-incompatible:** Add `cbar` option to `grid_archive_heatmap` (#193)
- **Backwards-incompatible:** Replace `get_random_elite()` with batched
  `sample_elites()` method (#192)
- **Backwards-incompatible:** Add EliteBatch and rename fields in Elite (#191)
- **Backwards-incompatible:** Rename bins to cells for consistency with
  literature (#189)
  - Archive constructors now take in `cells` argument instead of `bins`
  - Archive now have a `cells` property rather than a `bins` property
- **Backwards-incompatible:** Only use integer indices in archives (#185)
  - `ArchiveBase`
    - Replaced `storage_dims` (tuple of int) with `storage_dim` (int)
    - `_occupied_indices` is now a fixed-size array with `_num_occupied`
      indicating its current usage, and `_occupied_indices_cols` has been
      removed
    - `index_of` must now return an integer
