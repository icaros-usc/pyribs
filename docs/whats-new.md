# What's New in v0.7.0

The updates in v0.7.0 centered around making the archives more flexible and
adding new algorithmic features. Below we describe some of the key changes. For
the full list of changes, please refer to our [History page](./history).

## More Flexible Archives

We refactored our archives to build on a data structure we call an
{class}`~ribs.archives.ArrayStore`. An ArrayStore is essentially a dict mapping
from names ("fields") to fixed-size arrays. Archives store data like solutions,
objectives, and measures as fields in the ArrayStore. Building on ArrayStore
enabled us to create a more flexible API, which also meant introducing several
**breaking changes.** Below we list all the updates to the archives, ordered by
how likely they are to affect users.

### as_pandas() is deprecated in favor of data()

`archive.as_pandas()` has now been deprecated in favor of calling
`archive.data()`, which is a much more flexible method. Below are several
examples of the {meth}`~ribs.archives.ArchiveBase.data` method:

```python
# Returns a dict with all fields in the archive, e.g.,
#
# {
#   "solution": [[1.0, 1.0, ...], ...],
#   "objective": [1.5, ...],
#   "measures": [[1.0, 2.0], ...],
#   "threshold": [0.8, ...],
#   "index": [4, ...],
# }
archive.data()

# Returns a single array -- in this case, the shape will be (num elites,).
# We think this will be the most useful variant of data().
objective = archive.data("objective")

# Returns a dict with just the listed fields, e.g.,
#
# {
#   "objective": [1.5, ...],
#   "measures": [[1.0, 2.0], ...],
# }
archive.data(["objective", "measures"])

# Returns a tuple with just the listed fields, e.g.,
#
# (
#   [1.5, ...],
#   [[1.0, 2.0], ...],
# )
archive.data(["objective", "measures"], return_type="tuple")

# Returns an ArchiveDataFrame -- see below for several differences from the
# as_pandas ArchiveDataFrame.
archive.data(return_type="pandas")
```

In general, we believe users will find the single-field version (e.g.,
`archive.data("objective")` the most useful, with
`archive.data(return_type="pandas")` serving as a close replacement for
`as_pandas`. However, we note several differences in the ArchiveDataFrame
returned by `data()`:

1. Columns previously named `measure_X` are now named `measures_X` for
   consistency with other fields.
1. The columns are in a different order from before.
1. Iterating over an ArchiveDataFrame now returns a dict rather than the
   previous `Elite` namedtuple.
1. ArchiveDataFrame no longer has batch() methods. Instead, it has a get_field()
   method that converts columns back into their arrays, e.g.,
   `df.get_field("objective")`.

### Metadata has been removed in favor of custom archive fields

Previously, archives stored `metadata`, which were arbitrary objects associated
with each solution or elite. In pyribs 0.7.0, we have removed metadata and
instead support custom fields in archives. The example below shows how to use
custom fields -- pay attention to the `extra_fields` in the archive definition,
and the kwargs in `scheduler.tell()`.

```python
import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

archive = GridArchive(
    solution_dim=10,
    dims=[20, 20],
    ranges=[(-1, 1), (-1, 1)],
    # `extra_fields` is a dict mapping from "name" to a tuple of (shape, dtype).
    # Thus, extra_scalar is a scalar field of type float32, while extra_vector
    # is a length 10 vector field of type int32. This also works for other
    # archives.
    extra_fields={
        "extra_scalar": ((), np.float32),
        "extra_vector": ((10,), np.int32),
    },
)

# Emitter and scheduler definition -- feel free to skip over.
emitters = [
    EvolutionStrategyEmitter(
        archive,
        x0=[0.0] * 10,
        sigma0=0.1,
    ) for _ in range(3)
]
scheduler = Scheduler(archive, emitters)

solutions = scheduler.ask()

# The extra_fields become important in scheduler.tell(), when they must be
# passed in along with the usual objectives and measures. This also works for
# tell_dqd() in the case of DQD algorithms.
scheduler.tell(
    # The objective is the negative Sphere function, while the measures are the
    # first two coordinates of the 10D solution. Note that keyword arguments are
    # optional here (i.e., objective= and measures=).
    -np.sum(np.square(solutions), axis=1),
    solutions[:, :2],
    # The extra_fields specified in the archive must be passed in as kwargs.
    extra_scalar=solutions[:, 0],
    extra_vector=np.zeros((len(solutions), 10), dtype=np.int32),
)
```

Notably, it is possible to recover the original metadata behavior by defining a
`metadata` field as follows:

```python
archive = GridArchive(
    solution_dim=10,
    dims=[20, 20],
    ranges=[(-1, 1), (-1, 1)],
    extra_fields={
        "metadata": ((), object),
    },
)
```

### Additional Changes

#### retrieve() no longer returns EliteBatch

{meth}`~ribs.archives.ArchiveBase.retrieve` now returns a tuple of two objects:
(1) an `occupied` array indicating whether the given cells were occupied, and
(2) a dict containing the data of the elites in the given cells. Entries in the
dict are only valid if their corresponding cell was occupied. More info:
{pr}`414`.

#### Parameter names no longer include \_batch

Parameters for methods like {meth}`~ribs.archives.ArchiveBase.add` and
{meth}`~ribs.schedulers.Scheduler.tell` have been renamed to remove the `_batch`
suffix, as it is usually clear that we take in batch arguments. Methods that
require single arguments are already named with the `_single` suffix, e.g.,
`add_single` and `retrieve_single`.

#### Thresholds now included in elite data

The archive threshold is now included in
{attr}`~ribs.archives.ArchiveBase.best_elite` ({pr}`409`) and in data returned
by {meth}`~ribs.archives.ArchiveBase.retrieve` ({pr}`414`).

#### Elite and EliteBatch namedtuples are deprecated

The Elite and EliteBatch namedtuples have been removed, and methods will now
return dicts instead). This allows us to support custom field names. In
particular, iteration over an archive will now yield a dict instead of the Elite
namedtuple. More info: {pr}`397`

#### add() methods now return add_info dict

Instead of returning separate status and value arrays, the archive
{meth}`~ribs.archives.ArchiveBase.add` method now returns a dict that we refer
to as `add_info`. The `add_info` contains keys for `status` and `value` and may
contain further info in the future. Correspondingly, emitter methods like
{meth}`~ribs.emitters.EmitterBase.tell` now take in `add_info` instead of
separate `status_batch` and `value_batch` arguments. More info: {pr}`430`

## New Algorithmic Features

### Using pycma in Emitters

We added the {class}`~ribs.emitters.opt.PyCMAEvolutionStrategy` to support using
[pycma](https://github.com/CMA-ES/pycma) in emitters like the
{class}`~ribs.emitters.EvolutionStrategyEmitter`. The ES may be used by passing
`es="pycma_es"` to such emitters. Before using this, make sure that pycma is
installed, either by running `pip install cma` or `pip install ribs[pycma]`.

### New centroid generation methods in CVTArchive

Drawing from [Mouret 2023](https://dl.acm.org/doi/10.1145/3583133.3590726), we
now support alternative methods for generating centroids in
{class}`~ribs.archives.CVTArchive`. These methods may be specified via the
`centroid_method` parameter, for example:

```python
from ribs.archives import CVTArchive

archive = CVTArchive(
    solution_dim=10,
    cells=100,
    ranges=[(0.1, 0.5), (-0.6, -0.2)],
    # Alternatives: "kmeans" (default), "sobol", "scrambled_sobol", "halton"
    centroid_method="random",
)
```

### OMG-MEGA and OG-MAP-Elites

We have added the {class}`~ribs.emitters.GradientOperatorEmitter` to support the
OMG-MEGA and OG-MAP-Elites baseline algorithms from
[Fontaine 2021](https://arxiv.org/abs/2106.03894). The emitter may be used as
follows:

```python
from ribs.emitters import GradientOperatorEmitter

# For OMG-MEGA
GradientOperatorEmitter(
  sigma=0.0,
  sigma_g=10.0,
  measure_gradients=True,
  normalize_grad=True,
)

# For OG-MAP-Elites
GradientOperatorEmitter(
  sigma=0.5,
  sigma_g=0.5,
  measure_gradients=False,
  normalize_grad=False,
)
```
