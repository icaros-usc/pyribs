# What's New in v0.5.0

The new version of pyribs is here! Much has changed since our last version,
v0.4.0. Most of these changes can be categorized under two goals: (1) integrate
new algorithms via new emitters and schedulers, and (2) improve archive
performance via batched operations. We will review the changes related to these
goals and several other changes made to pyribs. For the full list of changes,
refer to our [History page](./history).

```{note}
To improve the pyribs API, many of these changes are backwards-incompatible and
break existing v0.4.0 code. In the future, we anticipate that we will introduce
fewer breaking changes as pyribs matures to have a clean, stable API.
```

## General

We'll start with some important general changes.

### Tutorials

We've added new tutorials and expanded our old ones! We recommend going over the
updated version of our introductory tutorial, {doc}`tutorials/lunar_lander`, to
get a feel for the new API. Then, learn how we have integrated the
[CMA-MAE](https://arxiv.org/abs/2205.10752) algorithm in
{doc}`tutorials/cma_mae`, and implement
[Differentiable Quality Diversity (DQD) algorithms](https://arxiv.org/abs/2106.03894)
in the tutorial {doc}`tutorials/tom_cruise_dqd`.

### Terminology

- **measures vs behaviors:** We have adopted the `measures` terminology of
  recent literature over the `behaviors` terminology in pyribs 0.4.0. Names such
  as `behavior_values` are now referred to as `measures`. While QD originated in
  neuroevolution with the purpose of producing diverse collections of agents, QD
  optimization has grown into a general-purpose optimization paradigm. For
  example, in an application where QD generates images of a face with varying
  age and hair length, it seems odd to refer to age and hair length as
  behaviors. Our new terminology instead refers to the age and hair length as
  measures of the solutions, where QD optimization must vary the outputs of
  those measures.
- **Batch arguments:** Many of our methods now operate in batch. The batch
  arguments are referred to as `solution_batch`, `objective_batch`,
  `measures_batch`, and `metadata_batch`, while individual arguments are
  referred to as `solution`, `objective`, `measures`, and `metadata`.
- **cells vs bins:** For consistency with the literature, we have replaced the
  term `bins` with `cells` when discussing archives.
- **optimizers are now schedulers:** To better reflect their role, the
  "optimizers" from pyribs v0.4.0 are now referred to as "schedulers". All
  schedulers are under the {mod}`ribs.schedulers` module, including
  {class}`~ribs.schedulers.Scheduler` and
  {class}`~ribs.schedulers.BanditScheduler`.

### Installation

There are now three distributions of pyribs, as shown in the table below. If you
do not use {mod}`ribs.visualize`, we recommend installing the default
distribution with with `pip install ribs` or `conda install pyribs`. If you do
use {mod}`ribs.visualize`, you can install pyribs with
`pip install ribs[visualize]` or `conda install pyribs-visualize`.

| Name      | PyPI Package      | Conda Package      | Description                                                                             |
| --------- | ----------------- | ------------------ | --------------------------------------------------------------------------------------- |
| Default   | `ribs`            | `pyribs`           | Basic pyribs package.                                                                   |
| Visualize | `ribs[visualize]` | `pyribs-visualize` | Adds visualization dependencies, currently just Matplotlib.                             |
| All       | `ribs[all]`       | `pyribs-all`       | Installs dependencies for all extra distributions (currently just the Visualize extra). |

## Goal 1: Implement New Algorithms

Our first major goal in pyribs v0.5.0 has been to implement an array of new
algorithms via new emitters and schedulers.

### Flexible CMA-ME and CMA-MAE with EvolutionStrategyEmitter

We introduce the {class}`~ribs.emitters.EvolutionStrategyEmitter` which replaces
the earlier `ImprovementEmitter`, `RandomDirectionEmitter`, and
`OptimizingEmitter`. This emitter may be used in both
[CMA-ME](https://arxiv.org/abs/1912.02400) and
[CMA-MAE](https://arxiv.org/abs/2205.10752). The behaviors of these earlier
emitters may be replicated by selecting an appropriate _ranker_ from the
{mod}`ribs.emitters.rankers` module. For example:

```python
EvolutionStrategyEmitter(archive, ..., ranker="2imp")  # Equivalent to ImprovementEmitter.
EvolutionStrategyEmitter(archive, ..., ranker="imp")  # Single-stage improvement ranking as is used in CMA-MAE.
EvolutionStrategyEmitter(archive, ..., ranker="2rd")  # Two-stage random direction ranking.
EvolutionStrategyEmitter(archive, ..., ranker="obj")  # Objective ranking as was done in OptimizingEmitter.
```

{class}`~ribs.emitters.EvolutionStrategyEmitter` also supports evolution
strategies other than CMA-ES, thus enabling it to implement scalable variants of
CMA-ME and CMA-MAE described in
[Tjanaka 2022](https://arxiv.org/abs/2210.02622). These evolution strategies are
available in the {mod}`ribs.emitters.opt` module. For example:

```python
EvolutionStrategyEmitter(archive, ..., es="sep_cma_es")  # sep-CMA-ES instead of CMA-ES.
```

### DQD Algorithms with GradientArborescenceEmitter

We have added a {class}`~ribs.emitters.GradientArborescenceEmitter` which
supports [DQD algorithms](https://arxiv.org/abs/2106.03894). For usage examples,
see the tutorial {doc}`tutorials/tom_cruise_dqd`.

### Custom Initial Solutions in Emitters

By default, on the first iteration (the first iteration is detected by checking
that the archive is empty), both {class}`~ribs.emitters.GaussianEmitter` and
{class}`~ribs.emitters.IsoLineEmitter` sample solutions from a Gaussian
distribution. However, many implementations of MAP-Elites sample solutions from
a uniform distribution on the first iteration. More generally, users may seek to
provide any custom population of solutions for the first iteration.

Before, it was possible to provide custom initial solutions by evaluating the
initial solutions and directly {meth}`~ribs.archives.ArchiveBase.add`'ing them
to the archive, like so:

```python
archive = ...
emitters = [GaussianEmitter(archive, ...)]
scheduler = Scheduler(archive, emitters)

initial_solutions = ...
objectives, measures = evaluate(initial_solutions)
archive.add(initial_solutions, objectives, measures)

for itr in range(1000):
    ...
```

However, it can be inconvenient to have to add this special case before the main
pyribs loop, e.g., if the evaluation function is more complex than a single
line. Thus, we now make it possible to pass the initial solutions to the
emitters so that on the first iteration of the QD algorithm (more specifically,
when the archive is empty), these initial solutions are returned.

```python
archive = ...
emitters = [GaussianEmitter(archive, ..., initial_solutions=[[0.0, 1.0], [1.3, 2.0]])]
scheduler = Scheduler(archive, emitters)

for itr in range(1000):
    # On the first iteration, `solutions` will be the `initial_solutions` that
    # were passed into GaussianEmitter.
    solutions = scheduler.ask()
```

### ME-MAP-Elites with BanditScheduler

We have added a new {class}`~ribs.schedulers.BanditScheduler` which maintains a
pool of emitters and only asks a subset of emitters for solutions on each
iteration. The emitters to ask are selected with a multi-armed bandit algorithm.
This scheduler is our implementation of the
[ME-MAP-Elites](https://arxiv.org/abs/2007.05352) algorithm.

## Goal 2: Improve Archive Performance via Batching

Before, pyribs archives operated on solutions one at a time. Now, most archives
(with the exception of SlidingBoundariesArchive) improve performance by relying
on batch operations.

### New Archive Methods

The following table shows the old methods and the names of the new methods which
operate on batched inputs. As a convenience, we have also included "single"
methods which can operate on single inputs.

| Old Method              | New Batched Method                               | New Single Method                                  |
| ----------------------- | ------------------------------------------------ | -------------------------------------------------- |
| `add()`                 | {meth}`~ribs.archives.ArchiveBase.add`           | {meth}`~ribs.archives.ArchiveBase.add_single`      |
| `get_index()`           | {meth}`~ribs.archives.ArchiveBase.index_of`      | {meth}`~ribs.archives.ArchiveBase.index_of_single` |
| `get_random_elite()`    | {meth}`~ribs.archives.ArchiveBase.sample_elites` | N/A                                                |
| `elite_with_behavior()` | {meth}`~ribs.archives.ArchiveBase.retrieve`      | {meth}`~ribs.archives.ArchiveBase.retrieve_single` |

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
- {meth}`~ribs.archives.ArchiveBase.add` formerly operated on single solutions.
  Now, it inserts solutions into the archive in batch. There is also an
  {meth}`~ribs.archives.ArchiveBase.add_single` which inserts solutions one at a
  time. The source code for {meth}`~ribs.archives.ArchiveBase.add_single` is
  more amenable to modifications than that of the batched
  {meth}`~ribs.archives.ArchiveBase.add` method.
- `get_random_elite()`, which sampled one elite from the archive, has been
  replaced with a batched {meth}`~ribs.archives.ArchiveBase.sample_elites`
  method which samples multiple elites at once.
- The `elite_with_behavior()` method which retrieved one elite with given
  measures has been replaced with {meth}`~ribs.archives.ArchiveBase.retrieve`,
  which retrieves a batch of such elites.

Some of these methods also include "single" versions which operate on single
solutions, e.g., {meth}`~ribs.archives.ArchiveBase.index_of_single` returns the
integer index of a single measures array.

### EliteBatch

We have added an {class}`~ribs.archives.EliteBatch` class to represent batches
of elites.

### ArchiveDataFrame Column Renaming

To reflect the terminology changes and the switch to integer indices in
archives, columns in {class}`~ribs.archives.ArchiveDataFrame` (returned by
{meth}`~ribs.archives.ArchiveBase.as_pandas`) are as follows:

| index | measure_0 | ... | objective | solution_0 | ... | metadata |
| ----- | --------- | --- | --------- | ---------- | --- | -------- |

Before, they were:

| index_0 | ... | behavior_0 | ... | objective | solution_0 | ... | metadata |
| ------- | --- | ---------- | --- | --------- | ---------- | --- | -------- |

We have also renamed the `ArchiveDataFrame` methods as follows:

| Old Name             | New Name                                                |
| -------------------- | ------------------------------------------------------- |
| `batch_behaviors()`  | {meth}`~ribs.archives.ArchiveDataFrame.measures_batch`  |
| `batch_indices()`    | {meth}`~ribs.archives.ArchiveDataFrame.index_batch`     |
| `batch_metadata()`   | {meth}`~ribs.archives.ArchiveDataFrame.metadata_batch`  |
| `batch_objectives()` | {meth}`~ribs.archives.ArchiveDataFrame.objective_batch` |
| `batch_solutions()`  | {meth}`~ribs.archives.ArchiveDataFrame.solution_batch`  |

## Miscellaneous

Finally, here are several miscellaneous improvements we have made to pyribs.

### Removal of initialize() Method for Archives

Previously, archives required calling `initialize(solution_dim)` after being
constructed. This method was typically called by the Optimizer/Scheduler. Now,
archives are directly constructed with the `solution_dim` argument for
simplicity.

```python
# v0.4.0 (OLD)
archive = GridArchive(...)
archive.initialize(solution_dim)  # Would be called by Optimizer / Scheduler.

# v0.5.0 (NEW)
archive = GridArchive(solution_dim, ...)
```

### More Flexible Visualization Tools

In all heatmap visualization tools, we have made the colorbar more flexible by
adding a `cbar` option to control which axes the colorbar appears on and a
`cbar_kwargs` option to pass arguments directly to Matplotlib's
{func}`~matplotlib.pyplot.colorbar`.

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
  heatmap, i.e., the ratio `height / width`.
- We now support heatmaps for 1D grid archives.

### Continuous Quality Diversity (CQD) Score

We support computing the
[Continuous Quality Diversity (CQD) Score](https://dl.acm.org/doi/10.1145/3520304.3534018)
in all archives with the {meth}`~ribs.archives.ArchiveBase.cqd_score` method.

### ArchiveStats

The {class}`~ribs.archives.ArchiveStats` object now includes the normalized QD
score (i.e., the QD score divided by the number of cells in the archive).
Furthermore, {class}`~ribs.archives.ArchiveStats` is now a dataclass rather than
a namedtuple.

```python
archive.stats.norm_qd_score  # Normalized QD score.
```

### Deprecation of Positional Arguments

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

### Input Validation

Many of our methods, e.g., the archives' {meth}`~ribs.archives.ArchiveBase.add`
method and schedulers' {meth}`~ribs.schedulers.Scheduler.ask` and
{meth}`~ribs.schedulers.Scheduler.tell` methods, now have input validation to
help catch common argument errors. For example, we check that arguments are the
correct shape, and we check that they do not have `NaN` or `inf` values.

### Deprecation of ribs.factory

We have removed `ribs.factory` as it is out of scope for the current library.
