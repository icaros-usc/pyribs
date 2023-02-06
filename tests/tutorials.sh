#!/bin/bash
# Runs minimal tutorials to make sure they are working end-to-end. Intended to
# be run from the root directory of the repo. By default, this script tests all
# tutorial notebooks, which takes a few minutes. Alternatively, just a single
# notebook can be tested by passing in the name of the notebook.
#
# Usage:
#   pip install .[visualize] jupyter nbconvert
#   bash tests/tutorials.sh  # Test all tutorials.
#   bash tests/tutorials.sh NOTEBOOK.ipynb  # Test a single notebook.

set -e  # Exit if any of the commands fail.
set -x  # Print out commands as they are run.

function test_notebook {
  TMP_FILE="tmp.ipynb"
  TMP_OUTPUT="tmp_output.ipynb"

  notebook="$1"
  echo "========== Testing $notebook =========="

  # Set the number of test iterations based on the tutorial name. Use this to
  # give special amounts to different tutorials if the default of 5 does not
  # work.
  case "$notebook" in
    tutorials/arm_repertoire.ipynb)
      test_itrs=50
      ;;
    tutorials/cma_mae.ipynb)
      test_itrs=5
      ;;
    *)
      test_itrs=3
      ;;
  esac
  echo "Test Iterations: ${test_itrs}"

  # Generate a copy of the notebook with reduced iterations.
  sed "s/total_itrs = [0-9]\\+/total_itrs = ${test_itrs}/g" < "$notebook" > "${TMP_FILE}"

  # Any further special replacements for testing.
  case "$notebook" in
    tutorials/arm_repertoire.ipynb)
      # Reduce samples so that CVTArchive runs quickly.
      sed -i 's/use_kd_tree=True,/use_kd_tree=True, samples=10000,/g' "${TMP_FILE}"
      ;;
    tutorials/lsi_mnist.ipynb)
      # Reduce data for the discriminator archive.
      sed -i 's/original_data = archive.as_pandas()/original_data = archive.as_pandas().loc[:5]/g' "${TMP_FILE}"
      ;;
  esac

  # Run the notebook. Timeout is long since some notebook cells take a while,
  # such as the ones that train the MNIST network.
  jupyter nbconvert \
    --to notebook \
    --execute "${TMP_FILE}" \
    --output "${TMP_OUTPUT}" \
    --ExecutePreprocessor.timeout=600
  rm -f "${TMP_FILE}" "${TMP_OUTPUT}"
}

if [ -z "$1" ]; then
  TUTORIALS=($(ls tutorials/*.ipynb))  # Contains all notebooks.
  for t in "${TUTORIALS[@]}"; do
    # Notebooks to exclude.
    case "$t" in
      # Example:
      # tutorials/tom_cruise_dqd.ipynb)
      #   continue
      #   ;;
    esac

    test_notebook "$t"
  done
else
  # If command line arg is passed in, test just that notebook.
  test_notebook "$1"
fi

echo "Success in $SECONDS seconds"
