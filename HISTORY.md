# History

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
