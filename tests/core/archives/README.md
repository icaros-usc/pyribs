# Steps for Testing an Archive

1. Add the name of your archive to `conftest.py` under `ARCHIVE_NAMES`.
1. Create data for it in `get_archive_data()`.
1. Parts of your archive should now be automatically tested in
   `archive_base_test.py`.
1. To test the rest of your archive, in particular the `add()` and `as_pandas()`
   methods, start a new test file for it. You may want to see an example such as
   `grid_archive_test.py`.
1. Just like the `_grid_data` fixture in `grid_archive_test.py`, create a
   fixture that uses `get_archive_data()` to retrieve test data. Feed that
   fixture into all your tests.
1. To benchmark your archive, view an example such as
   `grid_archive_benchmark.py`.
