# What's New in v0.8.0

Pyribs v0.8.0 adds support for several new algorithms while making it easier
than ever to design new ones. Below we highlight some of the most notable
updates. For the full list of changes, please refer to our History page.

## 🧠 New Algorithms

Pyribs v0.8.0 introduces implementations of several new algorithms:

- **Novelty Search**
  ([Lehman 2011](https://web.archive.org/web/20220707041732/https://eplex.cs.ucf.edu/papers/lehman_ecj11.pdf))
  via the {class}`~ribs.archives.ProximityArchive`. We illustrate how to run
  Novelty Search in our new tutorial, {doc}`tutorials/ns_maze`. Visualization is
  available via {func}`~ribs.visualize.proximity_archive_plot`.
  - Thanks to [@gresavage](https://github.com/gresavage) for helping with this
    implementation!
- **Density Descent Search**
  ([Lee 2024](https://dl.acm.org/doi/10.1145/3638529.3654001)) via the
  {class}`~ribs.archives.DensityArchive`. We illustrate how to run Density
  Descent Search in {doc}`examples/sphere`.
- **BOP-Elites**
  ([Kent 2024](https://ieeexplore.ieee.org/abstract/document/10472301)) via
  {class}`~ribs.emitters.BayesianOptimizationEmitter` and
  {class}`~ribs.schedulers.BayesianOptimizationScheduler`. An example of running
  BOP-Elites is available in the example here: {doc}`examples/bop_elites`.
  - As part of implementing BOP-Elites, a new
    {class}`~ribs.archives.GridArchive.retessellate` method has been added to
    GridArchive to allow changing the layout of the grid.
  - Thanks to [@szhaovas](https://github.com/szhaovas) for working on this
    implementation!

We have also added a {class}`~ribs.archives.CategoricalArchive` where the
measures can be categorical values, e.g., `["Cat", "Dog", "Mouse"]`.

The {doc}`supported-algorithms` page now includes a list of algorithms supported
in pyribs.

## 📜 Single-File Implementations for Archives

To make it easier to understand current archives and create new archives, we
have made each archive into a "single-file implementation." Previously,
{class}`~ribs.archives.ArchiveBase` contained a lot of archive logic. While this
made it easier to share code between archives, it also made it harder to
understand each archive's implementation. For instance, the implementation for
{class}`~ribs.archives.GridArchive` was split between the GridArchive file and
the ArchiveBase file. In addition, putting logic in ArchiveBase meant that new
implementations often had to override that logic, which quickly became
confusing. To overcome these issues, we have refactored ArchiveBase to be an
interface that contains almost no logic. All logic for archive implementations
is now placed in each archive's file, and to reduce repeated code, we have added
various utility functions. We anticipate that it will now be much easier to
understand how each archive operates.

## 🛠 Flexible Data Handling

Archives now support more flexible data specifications. Notably, solutions are
no longer restricted to 1D arrays. They can now be scalars or multi-dimensional
arrays; simply pass the appropriate shape as the
{attr}`~ribs.archives.ArchiveBase.solution_dim` in the archive. Separate data
types for the solutions, objectives, and measures are also supported by passing
in a dict as the `dtype` argument for archives.

For example, the following creates a GridArchive where the solutions can be
strings!

```python
archive = GridArchive(
    solution_dim=(),
    dims=[10, 20],
    ranges=[(-1, 1), (-2, 2)],
    dtype={
        "solution": object,
        "objective": np.float32,
        "measures": np.float32
    },
)

add_info = archive.add(
    solution=["This is Bob", "Bob says hi", "Good job Bob"],
    objective=[1.0, 2.0, 3.0],
    measures=[[0, 0], [0.25, 0.25], [0.5, 0.5]],
)
```

## 🚨 Breaking Changes

v0.8.0 includes several **backwards-incompatible changes**, most of which are
part of an ongoing effort to streamline the library:

- `archive.dtype` has been replaced with a more expressive `archive.dtypes`
  dictionary, since solutions, objectives and measures can now have different
  dtypes.
- {class}`~ribs.archives.cqd_score` is now a separate utility function, instead
  of being a method on the archives.
- The {attr}`~ribs.archives.ArchiveBase.field_list` and
  {attr}`~ribs.archives.ArchiveBase.dtypes` properties on current archives
  include the `index` field, since `index` is no longer a required part of
  archives.
- Archive `add` methods and the {class}`~ribs.archives.ArrayStore` have been
  simplified to no longer use "transforms," which significantly complicated the
  implementation.

## 🐛 Bug Fixes

In this release, we
[fixed a bug in BanditScheduler](https://github.com/icaros-usc/pyribs/pull/489)
thanks to [@Tekexa](https://github.com/Tekexa)! Now,
{class}`~ribs.schedulers.BanditScheduler` correctly maintains a stable number of
active emitters.

## Past Editions

Past editions of "What's New" are available below.

```{toctree}
:maxdepth: 1

whats-new/v0.7.0
whats-new/v0.5.0
```
