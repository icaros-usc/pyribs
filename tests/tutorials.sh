#!/bin/bash
# Runs minimal tutorials to make sure they are working properly. Intended to be
# run from the root directory of the repo. This script takes a few minutes to
# run.
#
# Usage:
#   pip install .[all] jupyterlab nbconvert
#   bash tests/tutorials.sh

set -e  # Exit if any of the commands fail.
set -x  # Print out commands as they are run.

TUTORIALS=($(ls examples/tutorials/lunar_lander.ipynb))
TMP_FILE="tmp.ipynb"
TMP_OUTPUT="tmp_output.ipynb"
for t in "${TUTORIALS[@]}"; do
  echo "========== Testing $t =========="
  sed 's/total_itrs = [0-9]\+/total_itrs = 5/g' < "$t" > "${TMP_FILE}"
  jupyter nbconvert --to notebook --execute "${TMP_FILE}" --output "${TMP_OUTPUT}"
  rm "${TMP_FILE}" "${TMP_OUTPUT}"
done

echo "Success in $SECONDS seconds"
