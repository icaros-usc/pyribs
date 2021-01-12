#!/bin/bash
# Runs minimal examples to make sure they are working properly. Intended to be
# run from the root directory of the repo. This script takes a while to run, so
# it is not intended to be run very often.
#
# Usage:
#   bash tests/examples.sh

set -e  # Exit if any of the commands fail.
set -x  # Print out commands as they are run.

TMPDIR="./examples_tmp"
if [ ! -d "${TMPDIR}" ]; then
  mkdir "${TMPDIR}"
fi

# run_sphere.py
SPHERE_OUTPUT="${TMPDIR}/run_sphere_output"
python examples/run_sphere.py map_elites 20 10000 "${SPHERE_OUTPUT}"
python examples/run_sphere.py line_map_elites 20 10000 "${SPHERE_OUTPUT}"
python examples/run_sphere.py cvt_map_elites 20 10000 "${SPHERE_OUTPUT}"
python examples/run_sphere.py line_cvt_map_elites 20 10000 "${SPHERE_OUTPUT}"

# Cleanup.
rm -rf $TMPDIR
echo "Success"
