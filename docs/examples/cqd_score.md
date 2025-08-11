# CQD Score

This script shows an example of how to use {func}`~ribs.archives.cqd_score` to
compute the Continuous QD Score metric. Introduced in
[Kent 2022](https://dl.acm.org/doi/10.1145/3520304.3534018), the CQD Score is a
metric that accounts for tradeoffs between objective values and distance to
desired measure values. Notably, the CQD Score does not depend on the
discretization/tessellation of the archive.

```{eval-rst}
.. literalinclude:: ../../examples/cqd_score.py
    :language: python
```
