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
  install_cmd=$(grep "pip install" "$1")
  $install_cmd
}

# Single-threaded for consistency.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# sphere.py
install_deps examples/sphere.py
SPHERE_OUTPUT="${TMPDIR}/sphere_output"
python examples/sphere.py map_elites --itrs 10 --outdir "${SPHERE_OUTPUT}"
python examples/sphere.py line_map_elites --itrs 10 --outdir "${SPHERE_OUTPUT}"

# CVT excluded since it takes a while to build the archive.
# python examples/sphere.py cvt_map_elites 10 "${SPHERE_OUTPUT}"
# python examples/sphere.py line_cvt_map_elites 10 "${SPHERE_OUTPUT}"

python examples/sphere.py cma_me_imp --itrs 10 --outdir "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_imp_mu --itrs 10 --outdir "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_rd --itrs 10 --outdir "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_rd_mu --itrs 10 --outdir "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_opt --itrs 10 --outdir "${SPHERE_OUTPUT}"
python examples/sphere.py cma_me_mixed --itrs 10 --outdir "${SPHERE_OUTPUT}"

python examples/sphere.py cma_mega --dim 20 --itrs 10 --outdir "${SPHERE_OUTPUT}"
python examples/sphere.py cma_mega_adam --dim 20 --itrs 10 --outdir "${SPHERE_OUTPUT}"

python examples/sphere.py cma_mae --dim 20 --itrs 10 --learning_rate 0.01 --outdir "${SPHERE_OUTPUT}"
python examples/sphere.py cma_maega --dim 20 --itrs 10 --learning_rate 0.01 --outdir "${SPHERE_OUTPUT}"

# lunar_lander.py
install_deps examples/lunar_lander.py
LUNAR_LANDER_OUTPUT="${TMPDIR}/lunar_lander_output"
python examples/lunar_lander.py --iterations 5 --outdir "${LUNAR_LANDER_OUTPUT}"

# Cleanup.
rm -rf $TMPDIR
echo "Success in $SECONDS seconds"
