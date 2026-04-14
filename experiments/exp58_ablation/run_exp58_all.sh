#!/bin/bash
set -e
DATASET_DIR=${1:-Dataset/NIR_cross_humidity}
RESULTS_ROOT=${2:-results/exp58_ablation}

SEEDS=(42 52 62 72 82)
VARIANTS=(full late_concat_only modulation_only no_residual)

for VARIANT in "${VARIANTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "============================================================"
    echo "Running variant=${VARIANT}, seed=${SEED}"
    echo "============================================================"
    python -m Ablation_study.train       --mode nir       --model hda_ablation       --dataset_dir "${DATASET_DIR}"       --seed "${SEED}"       --epochs 60       --ablation_variant "${VARIANT}"       --results_root "${RESULTS_ROOT}"       --exp "exp58_${VARIANT}_seed${SEED}"
  done
done

echo "All runs finished."
