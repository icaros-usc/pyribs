# Visualization Tests

This directory contains tests for ribs.visualize. For image comparison tests,
read
[these instructions](https://matplotlib.org/stable/devel/testing.html#writing-an-image-comparison-test).
Essentially, start by writing a test in one of the files in this directory;
let's pick `grid_archive_heatmap_test.py`. After writing this test, run it with
pytest, then go to the _root_ directory of this repo. There, you will find the
output image in `tests/visualize/baseline_images/grid_archive_heatmap_test`.
Copy this image into the `baseline_images/grid_archive_heatmap_test` directory
in this directory. Here's an example cp command:

```
cp result_images/grid_archive_heatmap_test/my_new_test.png \
    tests/baseline_images/grid_archive_heatmap_test
```

Assuming the output is as expected (and assuming the code is deterministic), the
test should now pass when it is re-run. The same applies for tests in other
files; for instance, you can do the same for `cvt_archive_heatmap_test`.
