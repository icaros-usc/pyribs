#!/bin/bash
# Lists tutorials for testing as a JSON list for GitHub Actions.
TUTORIALS=($(ls tutorials/*.ipynb tutorials/*/*.ipynb))
JSON_LIST=""
for notebook in "${TUTORIALS[@]}"; do
  case "$notebook" in
    # This is super flaky due to issues with swig and box2d.
    tutorials/lunar_lander.ipynb)
      continue
      ;;
    # Exclude certain notebooks that take too long.
    tutorials/tom_cruise_dqd.ipynb)
      continue
      ;;
    tutorials/qdhf.ipynb)
      continue
      ;;
    tutorials/qdaif.ipynb)
      continue
      ;;
    *)
      JSON_LIST="$JSON_LIST\"$notebook\","
      ;;
  esac
done
JSON_LIST=${JSON_LIST%","}  # Remove extra comma.
echo "matrix={\"tutorial\": [$JSON_LIST]}"
