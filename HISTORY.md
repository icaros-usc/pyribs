# History

## 0.4.0 (2021-07-19)

To learn about this release, see our blog post: https://pyribs.org/blog/0-4-0

### Changelog

#### API

- Add ribs.visualize.parallel_axes_plot for analyzing archives with high-dimensional BCs (#92)
- **Backwards-incompatible:** Reduce attributes and parameters in EmitterBase to make it easier to extend (#101)
- In Optimizer, support emitters that return any number of solutions in ask() (#101)
- **Backwards-incompatible:** Store metadata in archives as described in #87 (#103, #114, #115, #119)
- **Backwards-incompatible:** Rename "index" to "index_0" in CVTArchive.as_pandas for API consistency (#113)
- **Backwards-incompatible:** Make get_index() public in archives to emphasize each index's meaning (#128)
- **Backwards-incompatible:** Add index to get_random_elite() and elite_with_behavior() in archives (#129)
- Add clear() method to archive (#140, #146)
- Represent archive elites with an Elite namedtuple (#142)
- Add len and iter methods to archives (#151, #152)
- Add statistics to archives (#100, #157)
- Improve manipulation of elites by modifying as_pandas (#123, #149, #153, #158, #168)
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

- Move SlidingBoundariesArchive out of experimental by adding tests and fixing bugs (#93)
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
