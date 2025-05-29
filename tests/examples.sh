#!/bin/bash
# Runs minimal examples to make sure they are working properly. Intended to be
# run from the root directory of the repo. This script takes a few minutes to
# run.
#
# Usage:
#   bash tests/examples.sh

set -e  # Exit if any of the commands fail.
set -x  # Print out commands as they are run.

TMPDIR="./examples_tmp"
if [ ! -d "${TMPDIR}" ]; then
  mkdir "${TMPDIR}"
fi

function install_deps() {
  # Loop through all instances of `pip install` in the script and run the
  # installation commands.
  grep '^\s*pip install' "$1" | while read -r install_cmd ; do
      $install_cmd
  done
}

# Single-threaded for consistency.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

#
# sphere.py
#

install_deps examples/sphere.py
SPHERE_OUTPUT="${TMPDIR}/sphere_output"

# Read the list of algorithms in sphere.py into a bash array.
SPHERE_ALGOS=( $(cd examples && python -c "
from sphere import CONFIG
for algo in CONFIG:
  print(algo)
") )

# Test each algorithm.
for algo in "${SPHERE_ALGOS[@]}"; do
  # CVT excluded since it takes a while to build the archive.
  if [[ "$algo" != @(cvt_map_elites|line_cvt_map_elites) ]]; then
    python examples/sphere.py "$algo" --itrs 10 --outdir "${SPHERE_OUTPUT}"
  fi
done

#
# lunar_lander.py
#

install_deps examples/lunar_lander.py
LUNAR_LANDER_OUTPUT="${TMPDIR}/lunar_lander_output"
python examples/lunar_lander.py --iterations 5 --outdir "${LUNAR_LANDER_OUTPUT}"

# Cleanup.
rm -rf $TMPDIR
echo "Success in $SECONDS seconds"
