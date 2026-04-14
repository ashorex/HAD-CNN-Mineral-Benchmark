#!/bin/bash
set -e

DATASET_DIR=${1:-Dataset/NIR_cross_humidity}
RESULTS_ROOT=${2:-results/exp510_non_overlapping}

PRESETS=(split_A_current split_B_shifted split_C_shifted)
SEEDS=(42 52 62 72 82)

for PRESET in "${PRESETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "Running preset=${PRESET}, seed=${SEED}"
    echo "============================================================"
    python non_overlapping/exp510_train.py       --dataset_dir "${DATASET_DIR}"       --preset "${PRESET}"       --seed "${SEED}"       --results_root "${RESULTS_ROOT}"
  done
done

echo "All Experiment 5.10 runs finished."
