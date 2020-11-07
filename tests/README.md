# Tests

This directory contains tests and micro-benchmarks for ribs. The tests are
separated into two parts:

1. `core`: tests for the part of ribs that works with `pip install ribs`
2. `extras`: tests for the extra parts of ribs that only work when one installs
   the full ribs with `pip install ribs[all]`

To run these tests, you will also need to install the dev dependencies for ribs
with `pip install ribs[dev]` or `pip install -e .[dev]` (from the root directory
of the repo).
