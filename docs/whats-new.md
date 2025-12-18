# What's New in v0.9.0

We are excited to present pyribs 0.9.0, which introduces a host of new features
intended to make the library more flexible than ever! This release supports
Python 3.10 and up, with Python 3.9 being dropped due to being end-of-life.

## Overhauling CVTArchive

This release introduces a number of new features and (unfortunately) breaking
changes to {class}`~ribs.archives.CVTArchive`. For most users, it should suffice
to know that an initialization of CVTArchive that once looked like this:

```python
from ribs.archives import CVTArchive

archive = CVTArchive(
    solution_dim=12,
    cells=10000,
    ranges=[(-1, 1), (-1, 1)],
    use_kd_tree=True,
)
```

Now looks like this. Note that `cells` has been replaced with `centroids`, and
`use_kd_tree` has been replaced with `nearest_neighbors`.

```python
from ribs.archives import CVTArchive

archive = CVTArchive(
    solution_dim=12,
    centroids=10000,
    ranges=[(-1, 1), (-1, 1)],
    nearest_neighbors="scipy_kd_tree",
)
```

Our new tutorial serves as a starting point for using the new archive:
{doc}`/tutorials/cvt_centroids`.

### Detailed Changes

Below we provide further details on the changes, with even more detail available
in {issue}`621`.

- **Centroid generation has been moved out of CVTArchive.** In the past,
  CVTArchive contained methods for generating centroids that could be specified
  via the `centroid_method` parameter. However, centroid generation is extremely
  customizable, and we believe it should be left to the user rather than handled
  in the archive initialization itself. As such, **we have removed the
  `centroid_method` parameter as well as these centroid generation methods,**
  and we have instead added a new tutorial that shows different options for
  specifying and generating centroids: {doc}`/tutorials/cvt_centroids`.

  In a similar vein, it is now possible to generate centroids with the
  {func}`~ribs.archives.k_means_centroids` function, which samples points in a
  measure space and performs k-means clustering to identify the centroids. This
  function, as well as other workflows with centroids, are included in the above
  tutorial.

  Finally, since we have separated centroid generation from CVTArchive, **the
  `samples` property is now deprecated, as is the `plot_samples` parameter** in
  {func}`~ribs.visualize.cvt_archive_heatmap` and
  {func}`~ribs.visualize.cvt_archive_3d_plot`.

- **`cells` and `custom_centroids` are now replaced with the `centroids`
  parameter.** We found that `cells` and `custom_centroids` were redundant with
  each other. Instead, the new `centroids` parameter can either be an int
  indicating the number of cells in the archive, or an array-like with the
  coordinates of the centroids. This supports the separation of centroid
  generation from CVTArchive by making it easier to specify centroids.
  Previously, both `cells` and `custom_centroids` had to be specified to use
  centroids created by the user, but now, only `centroids` is needed.

- **Nearest-neighbor lookup is now more flexible.** Previously, we only
  supported two methods for nearest-neighbor lookup, via brute-force and via
  scipy's {class}`~scipy.spatial.KDTree`. To support more methods, **we have
  deprecated the `use_kd_tree` parameter and replaced it with a new
  `nearest_neighbors` parameter.** This `nearest_neighbors` parameter can be set
  to `"scipy_kd_tree"`, `"brute_force"`, or the newest option, `"sklearn_nn"`,
  which uses scikit-learn's {class}`~sklearn.neighbors.NearestNeighbors`.

  Correspondingly, we have added `sklearn_nn_kwargs` and `kdtree_query_kwargs`,
  which allow specifying more options to the internal objects used for
  nearest-neighbor search.

## New dtype Specifications for Archives

Previously, we found specifying separate dtypes for the solution, objective, and
measures in an archive to be complicated due to requiring passing a dict.
Instead, we now allow passing in individual dtypes: `solution_dtype`,
`objective_dtype`, and `measures_dtype`. It is also still possible to pass in
`dtype` to set all of these at once. For example, the following sets the
solutions to be objects, while the objective and measures are float32:

```python
import numpy as np

from ribs.archives import GridArchive

archive = GridArchive(
    solution_dim=(),
    dims=[20, 20],
    ranges=[(1, 10), (1, 10)],
    solution_dtype=object,
    objective_dtype=np.float32,
    measures_dtype=np.float32,
)
```

(This example is taken from the tutorial {doc}`tutorials/qdaif`.) For more info,
see {pr}`639`, {pr}`643`, and {pr}`661`.

## Multi-Dimensional Solutions in Emitters

This release improves support for multi-dimensional solutions in emitters. In
particular, the {class}`~ribs.emitters.GaussianEmitter` and
{class}`~ribs.emitters.IsoLineEmitter` can now generate multi-dimensional
solutions ({pr}`650`). In a similar vein, it is now possible to specify the
bounds of the solution space in emitters via `lower_bounds` and `upper_bounds`
({pr}`649`, {pr}`657`). The original `bounds` parameter is still supported but
cannot be used at the same time. Thus, the following creates a GaussianEmitter
that generates solutions of shape `(5, 5)` bounded by -1.0 and 1.0 in each
dimension:

```python
import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

archive = GridArchive(
    solution_dim=(5, 5),
    dims=[20, 20],
    ranges=[(-2, 2), (-2, 2)],
)

emitters = [
    GaussianEmitter(
        archive,
        sigma=0.1,
        x0=np.zeros((5, 5)),
        lower_bounds=np.full((5, 5), -1.0),
        upper_bounds=np.full((5, 5), 1.0),
        batch_size=64,
    )
]
```

## 🚨 Additional Breaking Changes

- In {class}`~ribs.archives.CVTArchive` and
  {class}`~ribs.archives.ProximityArchive`, `ckdtree_kwargs` is now referred to
  as `kdtree_kwargs`, as we have stopped using scipy's cKDTree in favor of
  KDTree ({pr}`676`)
- In {class}`~ribs.archives.ArrayStore`, `as_raw_dict` and `from_raw_dict` have
  been removed as they were not being used ({pr}`575`)

## ✨ Additional Features

- In the archives, {meth}`~ribs.archives.ArchiveBase.sample_elites` now supports
  the `replace` parameter to indicate whether elites should be replaced when
  sampling ({pr}`682`)
- {func}`~ribs.visualize.parallel_axes_plot` now supports plotting
  `ProximityArchive` ({pr}`647`)
- {class}`~ribs.archives.ArrayStore` now supports backends such as torch and
  cupy, drawing from the
  [Python array API standard](https://data-apis.org/array-api/latest/)
  ({issue}`570`, {pr}`645`)

## Developer Workflow

Finally, in addition to the above library improvements, we have migrated from
using yapf and pylint to using [ruff](https://docs.astral.sh/ruff/) and
[ty](https://docs.astral.sh/ty/) for formatting, linting, and type checking.

## Past Editions

Past editions of "What's New" are available below.

```{toctree}
:maxdepth: 1

whats-new/v0.8.0
whats-new/v0.7.0
whats-new/v0.5.0
```
