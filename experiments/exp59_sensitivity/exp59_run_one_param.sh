#!/bin/bash
set -e

PARAM=${1:-alpha}
DATASET_DIR=${2:-Dataset/NIR_cross_humidity}
RESULTS_ROOT=${3:-results/exp59_sensitivity}

if [ "$PARAM" = "alpha" ]; then
  VALUES=(1.0 1.5 2.0 2.5 3.0)
elif [ "$PARAM" = "delta" ]; then
  VALUES=(0.0 0.00075 0.0015 0.00225 0.0030)
elif [ "$PARAM" = "sigma0" ]; then
  VALUES=(0.4 0.6 0.8 1.0 1.2)
elif [ "$PARAM" = "eta" ]; then
  VALUES=(0.4 0.8 1.2 1.6 2.0)
elif [ "$PARAM" = "kappa" ]; then
  VALUES=(0.00 0.01 0.02 0.03 0.04)
else
  echo "Unsupported PARAM: $PARAM"
  exit 1
fi

SEEDS=(42 52 62 72 82)

for VALUE in "${VALUES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "Running param=${PARAM}, value=${VALUE}, seed=${SEED}"
    echo "============================================================"
    python -m Sensitivity.exp59_train       --dataset_dir "${DATASET_DIR}"       --param "${PARAM}"       --value "${VALUE}"       --seed "${SEED}"       --results_root "${RESULTS_ROOT}"
  done
done

echo "Finished PARAM=${PARAM}"
