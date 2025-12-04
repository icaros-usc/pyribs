#!/bin/bash
# Lists tutorials for testing as a JSON list for GitHub Actions.
TUTORIALS=($(ls tutorials/*.ipynb tutorials/*/*.ipynb))
JSON_LIST=""
for notebook in "${TUTORIALS[@]}"; do
  case "$notebook" in
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
