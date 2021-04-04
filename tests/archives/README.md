# Steps for Testing an Archive

1. Add the name of the archive to `conftest.py` under `ARCHIVE_NAMES`.
1. Create data for it in `get_archive_data()` in `conftest.py`.
1. Parts of the archive should now be automatically tested in
   `archive_base_test.py`.
1. To test the rest of the archive, in particular the `add()` method, start a
   new test file for it. For reference, see `grid_archive_test.py`.
1. Just like the `_data` fixture in `grid_archive_test.py`, create a
   fixture that uses `get_archive_data()` to retrieve test data. Feed that
   fixture into all the tests.
1. To benchmark the archive, view an example such as
   `grid_archive_benchmark.py`.
