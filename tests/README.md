# Tests

This directory contains tests and micro-benchmarks for pyribs. The tests mirror
the directory structure of `ribs`. To run these tests, install the dev
dependencies for ribs with `pip install ribs[dev]` or `pip install -e .[dev]`
(from the root directory of the repo).

For information on running tests, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## External Libraries

Some features of pyribs require external libraries that are optional and thus
not specified in the default installation command. We separate these tests into
separate directories:

- `visualize_qdax/` tests visualization of QDax components
- `emitters_pycma/` holds emitter tests that require pycma

## Additional Tests

This directory also contains:

- `examples.sh`: checks that the examples work end-to-end
- `tutorials.sh`: checks that the tutorials work end-to-end
