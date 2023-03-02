# History

## 0.5.1

This release contains miscellaneous edits to our documentation from v0.5.0.
There were no changes to library functionality in this release.

## 0.5.0

To learn about this release, see our page on What's New in v0.5.0:
https://docs.pyribs.org/en/stable/whats-new.html

### Changelog

#### API

- Schedulers warn if no solutions are inserted into archive (#320)
- Implement `BanditScheduler` (#299)
- **Backwards-incompatible:** Implement Scalable CMA-ES Optimizers (#274, #288)
- Make ribs.emitters.opt public (#281)
- Add normalized QD score to ArchiveStats (#276)
- **Backwards-incompatible:** Make ArchiveStats a dataclass (#275)
- **Backwards-incompatible:** Add shape checks to `tell()` and `tell_dqd()`
  methods (#269)
- Add method for computing CQD score in archives (#252)
- **Backwards-incompatible:** Deprecate positional arguments in constructors
  (#261)
- **Backwards-incompatible:** Allow custom initialization in Gaussian and
  IsoLine emitters (#259, #265)
- Implement CMA-MAE archive thresholds (#256, #260, #314)
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
- Re-run CVTArchive benchmarks and update CVTArchive (#235, #329)
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
- Raise ValueError for incorrect array shapes in archive methods (#219)
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

#### Bugs

- Fix boundary lines in sliding boundaries archive heatmap (#271)
- Fix negative eigenvalue in CMA-ES covariance matrix (#285)

#### Documentation

- Speed up lunar lander tutorial (#319)
- Add DQDTutorial (#267)
- Remove examples extra in favor of individual example deps (#306)
- Facilitate linking to latest version of documentation (#300)
- Update lunar lander tutorial with v0.5.0 features (#292)
- Improve tutorial and example overviews (#291)
- Move tutorials out of examples folder (#290)
- Update lunar lander to use Gymnasium (#289)
- Add CMA-MAE tutorial (#273, #284)
- Update README (#279)
- Add sphinx-codeautolink to docs (#206, #280)
- Fix documentation rendering issues on ReadTheDocs (#205)
- Fix typos and formatting in docstrings of `ribs/visualize.py` (#203)
- Add in-comment type hint rich linking (#204)
- Upgrade Sphinx dependencies (#202)

#### Improvements

- Move threadpoolctl from optimizer to CMA-ES (#241)
- Remove unnecessary emitter benchmarks (#231)
- Build docs during CI/CD workflow (#211)
- Drop Python 3.6 and add Python 3.10 support (#181)
- Add procedure for updating changelog (#182)
- Add 'visualize' extra (#183, #184, #302)

## 0.4.0 (2021-07-19)

To learn about this release, see our blog post: https://pyribs.org/blog/0-4-0

### Changelog

#### API

- Add ribs.visualize.parallel_axes_plot for analyzing archives with
  high-dimensional BCs (#92)
- **Backwards-incompatible:** Reduce attributes and parameters in EmitterBase to
  make it easier to extend (#101)
- In Optimizer, support emitters that return any number of solutions in ask()
  (#101)
- **Backwards-incompatible:** Store metadata in archives as described in #87
  (#103, #114, #115, #119)
- **Backwards-incompatible:** Rename "index" to "index_0" in
  CVTArchive.as_pandas for API consistency (#113)
- **Backwards-incompatible:** Make index_of() public in archives to emphasize
  each index's meaning (#128)
- **Backwards-incompatible:** Add index to get_random_elite() and
  elite_with_behavior() in archives (#129)
- Add clear() method to archive (#140, #146)
- Represent archive elites with an Elite namedtuple (#142)
- Add len and iter methods to archives (#151, #152)
- Add statistics to archives (#100, #157)
- Improve manipulation of elites by modifying as_pandas (#123, #149, #153, #158,
  #168)
- Add checks for optimizer array and list shapes (#166)

#### Documentation

- Add bibtex citations for tutorials (#122)
- Remove network training from Fooling MNIST tutorial (#161)
- Fix video display for lunar lander in Colab (#163)
- Fix Colab links in stable docs (#164)

#### Improvements

- Add support for Python 3.9 (#84)
- Test with pinned versions (#110)
- Increase minimum required versions for scipy and numba (#110)
- Refactor as_pandas tests (#114)
- Expand CI/CD to test examples and tutorials (#117)
- Tidy up existing tests (#120, #127)
- Fix vocab in various areas (#138)
- Fix dependency issues in tests (#139)
- Remove tox from CI (#143)
- Replace "entry" with "elite" in tests (#144)
- Use new archive API in ribs.visualize implementation (#155)

## 0.3.1 (2021-03-05)

This release features various bug fixes and improvements. In particular, we have
added tests for SlidingBoundariesArchive and believe it is ready for more
rigorous use.

### Changelog

- Move SlidingBoundariesArchive out of experimental by adding tests and fixing
  bugs (#93)
- Added nicer figures to the Sphere example with `grid_archive_heatmap` (#86)
- Added testing for Windows and MacOS (#83)
- Fixed package metadata e.g. description

## 0.3.0 (2021-02-05)

pyribs is now in beta. Since our alpha release (0.2.0), we have polished the
library and added new tutorials and examples to our documentation.

### Changelog

- Added a Lunar Lander example that extends the lunar lander tutorial (#70)
- Added New Tutorial: Illuminating the Latent Space of an MNIST GAN (#78)
- GridArchive: Added a boundaries attribute with the upper and lower bounds of
  each dimension's bins (#76)
- Fixed a bug where CMA-ME emitters do not work with float32 archives (#74)
- Fixed a bug where Optimizer is able to take in non-unique emitter instances
  (#75)
- Fixed a bug where GridArchive failed for float32 due to a small epsilon (#81)
- Fix issues with bounds in the SlidingBoundaryArchive (#77)
- Added clearer error messages for archives (#82)
- Modified the Python requirements to allow any version above 3.6.0 (#68)
- The wheel is now fixed so that it only supports py3 rather than py2 and py3
  (#68)
- Miscellaneous documentation fixes (#71)

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
