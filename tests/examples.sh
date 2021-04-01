#!/bin/bash
# Runs minimal examples to make sure they are working properly. Intended to be
# run from the root directory of the repo. This script takes a few minutes to
# run.
#
# Usage:
#   pip install .[all] .[examples]
#   bash tests/examples.sh

set -e  # Exit if any of the commands fail.
set -x  # Print out commands as they are run.

TMPDIR="./examples_tmp"
if [ ! -d "${TMPDIR}" ]; then
  mkdir "${TMPDIR}"
fi

# Single-threaded for consistency.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# sphere.py - CVT excluded since it takes a while to build the archive.
SPHERE_OUTPUT="${TMPDIR}/sphere_output"
python examples/sphere.py map_elites 20 10 "${SPHERE_OUTPUT}"
python examples/sphere.py line_map_elites 20 10 "${SPHERE_OUTPUT}"
# python examples/sphere.py cvt_map_elites 20 10 "${SPHERE_OUTPUT}"
# python examples/sphere.py line_cvt_map_elites 20 10 "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_imp 20 10 "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_imp_mu 20 10 "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_rd 20 10 "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_rd_mu 20 10 "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_opt 20 10 "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_mixed 20 10 "${SPHERE_OUTPUT}"

# lunar_lander.py
LUNAR_LANDER_OUTPUT="${TMPDIR}/lunar_lander_output"
python examples/lunar_lander.py --iterations 5 --outdir "${LUNAR_LANDER_OUTPUT}"

# Cleanup.
rm -rf $TMPDIR
echo "Success in $SECONDS seconds"
