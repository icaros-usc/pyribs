# History

## 0.8.2

This release focuses on improving documentation for pyribs, particularly by
adding the
[QDAIF tutorial](https://docs.pyribs.org/en/stable/tutorials/qdaif.html).

### Changelog

#### Documentation

- Migrate docs to sphinx-immaterial theme ({pr}`596`)
- Misc documentation fixes ({pr}`588`, {pr}`595`, {pr}`597`)
- Add QDAIF tutorial ({pr}`587`, {pr}`589`, {pr}`590`, {pr}`594`, {pr}`600`)
- Make all metrics start from values in archive stats ({pr}`586`)
- Update dependencies for lunar lander ({pr}`585`)

#### Improvements

- Migrate from yapf to ruff for formatting ({pr}`581`, {pr}`582`, {pr}`583`)

## 0.8.1

This release makes some minor updates to 0.8.0.

### Changelog

#### API

- **Backwards-incompatible:** Remove convenience extras ribs[pymoo] and
  ribs[pycma] ({pr}`567`)

#### Documentation

- Update Discord and X links in README ({pr}`566`)

## 0.8.0

To learn about this release, see our page on What's New in v0.8.0:
https://docs.pyribs.org/en/stable/whats-new.html

### Changelog

#### API

- Implement BOP-Elites by adding BayesianOptimizationEmitter and
  BayesianOptimizationScheduler; add BOP-Elites demo ({pr}`496`)
- Allow result_archive to be positional arg in schedulers ({pr}`557`)
- **Backwards-incompatible:** Make `num_active` a keyword-only arg in
  BanditScheduler ({pr}`557`)
- Add CategoricalArchive ({pr}`549`)
- Support solutions with non-1D shapes ({pr}`550`)
- **Backwards-incompatible:** Move cqd_score into a separate function
  ({pr}`537`)
- **Backwards-incompatible:** Make archive field_list and dtypes props include
  index ({pr}`532`)
- **Backwards-incompatible:** Remove thresholds from SlidingBoundariesArchive
  ({pr}`527`)
- **Backwards-incompatible:** Remove transforms from archive add operations
  ({pr}`525`)
- Add retessellate method to GridArchive ({pr}`516`)
- **Backwards-incompatible:** Tidy up operator implementation ({pr}`507`,
  {pr}`510`)
- Drop Python 3.8 support and remove pinned requirements {{pr}`497`)
- **Backwards-incompatible:** BanditScheduler: Add emitter_pool and active attr;
  remove emitters attr ({pr}`494`)
- Add DensityArchive and DensityRanker for Density Descent Search ({pr}`483`,
  {pr}`504`, {pr}`487`, {pr}`543`, {pr}`546`)
- Add NoveltyRanker for novelty search ({pr}`477`)
- Add proximity_archive_plot for visualizing ProximityArchive ({pr}`476`,
  {pr}`480`, {pr}`523`)
- Support novelty search with local competition in ProximityArchive ({pr}`481`)
- Add ProximityArchive for novelty search ({pr}`472`, {pr}`479`, {pr}`484`,
  {pr}`521`, {pr}`527`)
- Support diversity optimization in Scheduler.tell ({pr}`473`)
- Allow specifying separate dtypes for solution, objective, and measures
  ({pr}`471`)
- Replace archive.dtype with archive.dtypes dict that holds dtype of every field
  ({pr}`470`)

#### Bugs

- Make emitter bounds dtype match solution dtype ({pr}`519`)
- Fix `BanditScheduler` behaviour: the number of active emitters remains stable
  ({pr}`489`)

#### Documentation

- Add Novelty Search with Kheperax tutorial ({pr}`552`)
- Add Supported Algorithms page ({pr}`559`)
- Add cqd_score example ({pr}`537`)
- Update sphere example for consistency ({pr}`505`)
- Tutorial edits ({pr}`500`, {pr}`553`, {pr}`554`)
- Add version selector to docs ({pr}`495`)
- Update gymnasium and lunar lander version ({pr}`493`)
- Add tutorial page on Optuna integration ({pr}`492`)
- Switch from std to var in arm tutorial ({pr}`486`)
- Fix documentation conf for readthedocs deprecations ({pr}`485`)
- Add novelty search with CMA-ES to sphere example ({pr}`478`, {pr}`482`)
- Clarify errors in scheduler docstrings ({pr}`488`)

#### Improvements

- Remove `np_scalar` util by making archive dtypes be numpy scalar types
  ({pr}`534`)
- Refactor archives into single-file implementations ({pr}`518`, {pr}`521`,
  {pr}`526`, {pr}`528`, {pr}`529`, {pr}`530`, {pr}`533`, {pr}`535`)
- Make ArrayStore.data return ArchiveDataFrame instead of DataFrame ({pr}`522`)
- Migrate to pyproject.toml ({pr}`514`)
- Set vmin and vmax to None if archive is empty in ribs.visualize ({pr}`513`,
  {pr}`523`)
- Remove operators from GaussianEmitter and IsoLineEmitter ({pr}`508`)
- Update QDax visualizations to match QDax 0.5.0 ({pr}`502`)
- Skip qdax tests if qdax not installed ({pr}`491`)
- Move yapf after isort in pre-commit ({pr}`490`)
- Remove `_cells` attribute from ArchiveBase ({pr}`475`)
- Upgrade setup-miniconda to v3 due to deprecation ({pr}`464`)

## 0.7.1

This release introduces the
[QDHF tutorial](https://docs.pyribs.org/en/stable/tutorials/qdhf.html)! It also
makes a couple of minor usability improvements, such as better error checking.

### Changelog

#### API

- Support Python 3.12 ({pr}`390`)

#### Improvements

- Add qd score to lunar lander example ({pr}`458`)
- Raise error if `result_archive` and `archive` have different fields
  ({pr}`461`)
- Warn user if resampling for bounds takes too long in ESs ({pr}`462`)

#### Documentation

- Add QDHF tutorial ({pr}`459`)

#### Bugs

- Fix solution retrieval in lunar lander eval ({pr}`457`)

## 0.7.0

To learn about this release, see our page on What's New in v0.7.0:
https://docs.pyribs.org/en/stable/whats-new/v0.7.0.html

### Changelog

#### API

- Add GeneticAlgorithmEmitter with Internal Operator Support ({pr} `427`)
- Support alternative centroid generation methods in CVTArchive ({pr}`417`,
  {pr}`437`)
- Add PyCMAEvolutionStrategy for using pycma in ES emitters ({pr}`434`)
- **Backwards-incompatible:** Add ranking values to evolution strategy tell
  method ({pr}`438`)
- **Backwards-incompatible:** Move evolution strategy bounds to init ({pr}`436`)
- **Backwards-incompatible:** Use seed instead of rng in ranker ({pr}`432`)
- **Backwards-incompatible:** Replace status and value with add_info ({pr}`430`)
- Support custom data fields in archive, emitters, and scheduler ({pr}`421`,
  {pr}`429`)
- **Backwards-incompatible:** Remove `_batch` from parameter names ({pr}`422`,
  {pr}`424`, {pr}`425`, {pr}`426`, {pr}`428`)
- Add Gaussian, IsoLine Operators and Refactor GaussianEmitter/IsoLineEmitter
  ({pr}`418`)
- **Backwards-incompatible:** Remove metadata in favor of custom fields
  ({pr}`420`)
- Add Base Operator Interface and Emitter Operator Retrieval ({pr}`416`)
- **Backwards-incompatible:** Return occupied booleans in retrieve ({pr}`414`)
- **Backwards-incompatible:** Deprecate `as_pandas` in favor of
  `data(return_type="pandas")` ({pr}`408`)
- **Backwards-incompatible:** Replace ArchiveDataFrame batch methods with
  `get_field` ({pr}`413`)
- Add field_list and data methods to archives ({pr}`412`)
- Include threshold in `archive.best_elite` ({pr}`409`)
- **Backwards-incompatible:** Replace Elite and EliteBatch with dicts
  ({pr}`397`)
- **Backwards-incompatible:** Rename `measure_*` columns to `measures_*` in
  `as_pandas` ({pr}`396`)
- Add ArrayStore data structure ({pr}`395`, {pr}`398`, {pr}`400`, {pr}`402`,
  {pr}`403`, {pr}`404`, {pr}`406`, {pr}`407`, {pr}`411`)
- Add GradientOperatorEmitter to support OMG-MEGA and OG-MAP-Elites ({pr}`348`)

#### Improvements

- Raise error when threshold_min is set but learning_rate is not ({pr}`453`)
- Fix interval_size in CVTArchive and SlidingBoundariesArchive ({pr}`452`)
- Allow overriding ES in sphere example ({pr}`439`)
- Use NumPy SeedSequence in emitters ({pr}`431`, {pr}`440`)
- Use numbers types when checking arguments ({pr}`419`)
- Reimplement ArchiveBase using ArrayStore ({pr}`399`)
- Use chunk computation in CVT brute force calculation to reduce memory usage
  ({pr}`394`)
- Test pyribs installation in tutorials ({pr}`384`)
- Add cron job for testing installation ({pr}`389`, {pr}`401`)
- Fix broken cross-refs in docs ({pr}`393`)

#### Documentation

- Tidy up LSI MNIST notebook ({pr}`444`)

## 0.6.4

Small release that adds the scalable CMA-MAE tutorial.

### Changelog

#### Documentation

- Add tutorial on scalable CMA-MAE variants ({pr}`433`, {pr}`443`)

## 0.6.3

Small patch release due to deprecation issues.

### Changelog

#### Improvements

- Replace np.product with np.prod due to deprecation ({pr}`385`)

## 0.6.2

Small patch release due to installation issues in our tutorials.

### Changelog

#### API

- Import ribs[visualize] in tutorials that need it ({pr}`379`)

#### Improvements

- Switch to a branch-based release model ({pr}`382`)

## 0.6.1

(This release was removed)

## 0.6.0

### Changelog

#### API

- Drop Python 3.7 support and upgrade dependencies ({pr}`350`)
- Add visualization of QDax repertoires ({pr}`353`)
- Improve cvt_archive_heatmap flexibility ({pr}`354`)
- Clip Voronoi regions in cvt_archive_heatmap ({pr}`356`)
- **Backwards-incompatible:** Allow using kwargs for colorbar in
  parallel_axes_plot ({pr}`358`)
  - Removes cbar_orientaton and cbar_pad args for parallel_axes_plot
- Add `rasterized` arg for heatmaps ({pr}`359`)
- Support 1D cvt_archive_heatmap ({pr}`362`)
- Add 3D plots for CVTArchive ({pr}`371`)
- Add visualization of 3D QDax repertoires ({pr}`373`)
- Enable plotting custom data in visualizations ({pr}`374`)

#### Documentation

- Use dask instead of multiprocessing for lunar lander tutorial ({pr}`346`)
- pip install swig before gymnasium[box2d] in lunar lander tutorial ({pr}`346`)
- Fix lunar lander dependency issues ({pr}`366`, {pr}`367`)
- Simplify DQD tutorial imports ({pr}`369`)
- Improve visualization docs examples ({pr}`372`)

#### Improvements

- Improve developer workflow with pre-commit ({pr}`351`, {pr}`363`)
- Speed up 2D cvt_archive_heatmap by order of magnitude ({pr}`355`)
- Refactor visualize module into multiple files ({pr}`357`)
- Refactor visualize tests into multiple files ({pr}`370`)
- Add GitHub link roles in documentation ({pr}`361`)
- Refactor argument validation utilities ({pr}`365`)
- Use Conda envs in all CI jobs ({pr}`368`)
- Split tutorial CI into multiple jobs ({pr}`375`)

## 0.5.2

This release contains miscellaneous edits to our documentation from v0.5.1.
Furthermore, the library is updated to support Python 3.11, removed deprecated
options, and strengthened with more robust checks and error messages in the
schedulers.

### Changelog

#### API

- Support Python 3.11 ({pr}`342`)
- Check that emitters passed in are lists/iterables in scheduler ({pr}`341`)
- Fix Matplotlib `get_cmap` deprecation ({pr}`340`)
- **Backwards-incompatible:** Default `plot_centroids` to False when plotting
  ({pr}`339`)
- Raise error messages when `ask` is called without `ask_dqd` ({pr}`338`)

#### Documentation

- Add BibTex citation for GECCO 2023 ({pr}`337`)

#### Improvements

- Update distribution dependencies ({pr}`344`)

## 0.5.1

This release contains miscellaneous edits to our documentation from v0.5.0.
There were no changes to library functionality in this release.

## 0.5.0

To learn about this release, see our page on What's New in v0.5.0:
https://docs.pyribs.org/en/stable/whats-new/v0.5.0.html

### Changelog

#### API

- Schedulers warn if no solutions are inserted into archive ({pr}`320`)
- Implement `BanditScheduler` ({pr}`299`)
- **Backwards-incompatible:** Implement Scalable CMA-ES Optimizers ({pr}`274`,
  {pr}`288`)
- Make ribs.emitters.opt public ({pr}`281`)
- Add normalized QD score to ArchiveStats ({pr}`276`)
- **Backwards-incompatible:** Make ArchiveStats a dataclass ({pr}`275`)
- **Backwards-incompatible:** Add shape checks to `tell()` and `tell_dqd()`
  methods ({pr}`269`)
- Add method for computing CQD score in archives ({pr}`252`)
- **Backwards-incompatible:** Deprecate positional arguments in constructors
  ({pr}`261`)
- **Backwards-incompatible:** Allow custom initialization in Gaussian and
  IsoLine emitters ({pr}`259`, {pr}`265`)
- Implement CMA-MAE archive thresholds ({pr}`256`, {pr}`260`, {pr}`314`)
  - Revive the old implementation of `add_single` removed in ({pr}`221`)
  - Add separate tests for `add_single` and `add` with single solution
- Fix all examples and tutorials ({pr}`253`)
- Add restart timer to `EvolutionStrategyEmitter` and
  `GradientArborescenceEmitter`({pr}`255`)
- Rename fields and update documentation ({pr}`249`, {pr}`250`)
  - **Backwards-incompatible:** rename `Optimizer` to `Scheduler`
  - **Backwards-incompatible:** rename `objective_value` to `objective`
  - **Backwards-incompatible:** rename `behavior_value`/`bcs` to `measures`
  - **Backwards-incompatible:** `behavior_dim` in archives is now `measure_dim`
  - Rename `n_solutions` to `batch_size` in `Scheduler`.
- Add `GradientArborescenceEmitter`, which is used to implement CMA-MEGA
  ({pr}`240`, {pr}`263`, {pr}`264`, {pr}`282`, {pr}`321`)
- Update emitter `tell()` docstrings to no longer say "Inserts entries into
  archive" ({pr}`247`)
- Expose `emitter.restarts` as a property ({pr}`248`)
- Specify that `x0` is 1D for all emitters ({pr}`244`)
- Add `best_elite` property for archives ({pr}`237`)
- Rename methods in ArchiveDataFrame and rename as_pandas behavior columns
  ({pr}`236`)
- Re-run CVTArchive benchmarks and update CVTArchive ({pr}`235`, {pr}`329`)
  - **Backwards-incompatible:** `use_kd_tree` now defaults to True since the k-D
    tree is always faster than brute force in benchmarks.
- Allow adding solutions one at a time in optimizer ({pr}`233`)
- Minimize numba usage ({pr}`232`)
- **Backwards-incompatible:** Implement batch addition in archives ({pr}`221`,
  {pr}`242`)
  - `add` now adds a batch of solutions to the archive
  - `add_single` adds a single solution
- `emitter.tell` now takes in `status_batch` and `value_batch` ({pr}`227`)
- Make epsilon configurable in archives ({pr}`226`)
- **Backwards-incompatible:** Remove ribs.factory ({pr}`225`, {pr}`228`)
- **Backwards-incompatible:** Replaced `ImprovementEmitter`,
  `RandomDirectionEmitter`, and `OptimizingEmitter` with
  `EvolutionStrategyEmitter` ({pr}`220`, {pr}`223`, {pr}`278`)
- Raise ValueError for incorrect array shapes in archive methods ({pr}`219`)
- Introduced the Ranker object, which is responsible for ranking the solutions
  based on different objectives ({pr}`209`, {pr}`222`, {pr}`245`)
- Add index_of_single method for getting index of measures for one solution
  ({pr}`214`)
- **Backwards-incompatible:** Replace elite_with_behavior with retrieve and
  retrieve_single in archives ({pr}`213`, {pr}`215`, {pr}`295`)
- **Backwards-incompatible:** Replace get_index with batched index_of method in
  archives ({pr}`208`)
  - Also added `grid_to_int_index` and `int_to_grid_index` methods for
    `GridArchive` and `SlidingBoundariesArchive`
- **Backwards-incompatible:** Made it such that each archive is initialized
  fully in its constructor instead of needing a separate
  .initialize(solution_dim) call ({pr}`200`)
- **Backwards-incompatible:** Add `sigma`, `sigma0` options to
  `gaussian_emitter` and `iso_line_emitter` ({pr}`199`)
  - `gaussian_emitter` constructor requires `sigma`; `sigma0` is optional.
  - `iso_line_emitter` constructor takes in optional parameter `sigma0`.
- **Backwards-incompatible:** Add `cbar`, `aspect` options for
  `cvt_archive_heatmap` ({pr}`197`)
- **Backwards-incompatible:** Add `aspect` option to `grid_archive_heatmap` +
  support for 1D heatmaps ({pr}`196`)
  - `square` option no longer works
- **Backwards-incompatible:** Add `cbar` option to `grid_archive_heatmap`
  ({pr}`193`)
- **Backwards-incompatible:** Replace `get_random_elite()` with batched
  `sample_elites()` method ({pr}`192`)
- **Backwards-incompatible:** Add EliteBatch and rename fields in Elite
  ({pr}`191`)
- **Backwards-incompatible:** Rename bins to cells for consistency with
  literature ({pr}`189`)
  - Archive constructors now take in `cells` argument instead of `bins`
  - Archive now have a `cells` property rather than a `bins` property
- **Backwards-incompatible:** Only use integer indices in archives ({pr}`185`)
  - `ArchiveBase`
    - Replaced `storage_dims` (tuple of int) with `storage_dim` (int)
    - `_occupied_indices` is now a fixed-size array with `_num_occupied`
      indicating its current usage, and `_occupied_indices_cols` has been
      removed
    - `index_of` must now return an integer

#### Bugs

- Fix boundary lines in sliding boundaries archive heatmap ({pr}`271`)
- Fix negative eigenvalue in CMA-ES covariance matrix ({pr}`285`)

#### Documentation

- Speed up lunar lander tutorial ({pr}`319`)
- Add DQDTutorial ({pr}`267`)
- Remove examples extra in favor of individual example deps ({pr}`306`)
- Facilitate linking to latest version of documentation ({pr}`300`)
- Update lunar lander tutorial with v0.5.0 features ({pr}`292`)
- Improve tutorial and example overviews ({pr}`291`)
- Move tutorials out of examples folder ({pr}`290`)
- Update lunar lander to use Gymnasium ({pr}`289`)
- Add CMA-MAE tutorial ({pr}`273`, {pr}`284`)
- Update README ({pr}`279`)
- Add sphinx-codeautolink to docs ({pr}`206`, {pr}`280`)
- Fix documentation rendering issues on ReadTheDocs ({pr}`205`)
- Fix typos and formatting in docstrings of `ribs/visualize.py` ({pr}`203`)
- Add in-comment type hint rich linking ({pr}`204`)
- Upgrade Sphinx dependencies ({pr}`202`)

#### Improvements

- Move threadpoolctl from optimizer to CMA-ES ({pr}`241`)
- Remove unnecessary emitter benchmarks ({pr}`231`)
- Build docs during CI/CD workflow ({pr}`211`)
- Drop Python 3.6 and add Python 3.10 support ({pr}`181`)
- Add procedure for updating changelog ({pr}`182`)
- Add 'visualize' extra ({pr}`183`, {pr}`184`, {pr}`302`)

## 0.4.0 (2021-07-19)

To learn about this release, see our blog post: https://pyribs.org/blog/0-4-0

### Changelog

#### API

- Add ribs.visualize.parallel_axes_plot for analyzing archives with
  high-dimensional BCs ({pr}`92`)
- **Backwards-incompatible:** Reduce attributes and parameters in EmitterBase to
  make it easier to extend ({pr}`101`)
- In Optimizer, support emitters that return any number of solutions in ask()
  ({pr}`101`)
- **Backwards-incompatible:** Store metadata in archives as described in
  {pr}`87` ({pr}`103`, {pr}`114`, {pr}`115`, {pr}`119`)
- **Backwards-incompatible:** Rename "index" to "index_0" in
  CVTArchive.as_pandas for API consistency ({pr}`113`)
- **Backwards-incompatible:** Make index_of() public in archives to emphasize
  each index's meaning ({pr}`128`)
- **Backwards-incompatible:** Add index to get_random_elite() and
  elite_with_behavior() in archives ({pr}`129`)
- Add clear() method to archive ({pr}`140`, {pr}`146`)
- Represent archive elites with an Elite namedtuple ({pr}`142`)
- Add len and iter methods to archives ({pr}`151`, {pr}`152`)
- Add statistics to archives ({pr}`100`, {pr}`157`)
- Improve manipulation of elites by modifying as_pandas ({pr}`123`, {pr}`149`,
  {pr}`153`, {pr}`158`, {pr}`168`)
- Add checks for optimizer array and list shapes ({pr}`166`)

#### Documentation

- Add bibtex citations for tutorials ({pr}`122`)
- Remove network training from Fooling MNIST tutorial ({pr}`161`)
- Fix video display for lunar lander in Colab ({pr}`163`)
- Fix Colab links in stable docs ({pr}`164`)

#### Improvements

- Add support for Python 3.9 ({pr}`84`)
- Test with pinned versions ({pr}`110`)
- Increase minimum required versions for scipy and numba ({pr}`110`)
- Refactor as_pandas tests ({pr}`114`)
- Expand CI/CD to test examples and tutorials ({pr}`117`)
- Tidy up existing tests ({pr}`120`, {pr}`127`)
- Fix vocab in various areas ({pr}`138`)
- Fix dependency issues in tests ({pr}`139`)
- Remove tox from CI ({pr}`143`)
- Replace "entry" with "elite" in tests ({pr}`144`)
- Use new archive API in ribs.visualize implementation ({pr}`155`)

## 0.3.1 (2021-03-05)

This release features various bug fixes and improvements. In particular, we have
added tests for SlidingBoundariesArchive and believe it is ready for more
rigorous use.

### Changelog

- Move SlidingBoundariesArchive out of experimental by adding tests and fixing
  bugs ({pr}`93`)
- Added nicer figures to the Sphere example with `grid_archive_heatmap`
  ({pr}`86`)
- Added testing for Windows and MacOS ({pr}`83`)
- Fixed package metadata e.g. description

## 0.3.0 (2021-02-05)

pyribs is now in beta. Since our alpha release (0.2.0), we have polished the
library and added new tutorials and examples to our documentation.

### Changelog

- Added a Lunar Lander example that extends the lunar lander tutorial ({pr}`70`)
- Added New Tutorial: Illuminating the Latent Space of an MNIST GAN ({pr}`78`)
- GridArchive: Added a boundaries attribute with the upper and lower bounds of
  each dimension's bins ({pr}`76`)
- Fixed a bug where CMA-ME emitters do not work with float32 archives ({pr}`74`)
- Fixed a bug where Optimizer is able to take in non-unique emitter instances
  ({pr}`75`)
- Fixed a bug where GridArchive failed for float32 due to a small epsilon
  ({pr}`81`)
- Fix issues with bounds in the SlidingBoundaryArchive ({pr}`77`)
- Added clearer error messages for archives ({pr}`82`)
- Modified the Python requirements to allow any version above 3.6.0 ({pr}`68`)
- The wheel is now fixed so that it only supports py3 rather than py2 and py3
  ({pr}`68`)
- Miscellaneous documentation fixes ({pr}`71`)

## 0.2.0 (2021-01-29)

- Alpha release

## 0.2.1 (2021-01-29)

- Package metadata fixes (author, email, url)
- Miscellaneous documentation improvements

## 0.1.1 (2021-01-29)

- Test release (now removed)

## 0.1.0 (2020-09-11)

- Test release (now removed)

## 0.0.0 (2020-09-11)

- pyribs begins
