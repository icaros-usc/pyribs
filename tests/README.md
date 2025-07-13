# Tests

This directory contains tests and micro-benchmarks for pyribs. The tests mirror
the directory structure of `ribs`. To run these tests, install the dev
dependencies for ribs with `pip install ribs[dev]` or `pip install -e .[dev]`
(from the root directory of the repo).

For information on running tests, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## External Libraries

Some features of pyribs require external libraries that are optional and thus
not specified in the default installation command. We place these tests into
separate directories, such as `visualize_qdax/` and `emitters_pycma/`.

## Additional Tests

This directory also contains:

- `examples.sh`: checks that the examples work end-to-end
- `tutorials.sh`: checks that the tutorials work end-to-end

## Array API

To write tests for components that feature the Array API, use the
`xp_and_device` fixture to receive a tuple with the array namespace `xp` and the
device `device`. `xp_and_device` is implemented in `tests/conftest.py`.
