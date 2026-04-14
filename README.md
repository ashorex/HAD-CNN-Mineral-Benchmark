# HAD-CNN-Mineral-Benchmark

A reproducible benchmark-oriented framework for humidity-aware mineral spectral classification using the USGS Spectral Library.

## Included benchmark settings

### Experiment 1 (matched humidity)
- train humidities: `np.linspace(0, 1, 30)`
- test humidities: `np.arange(0.1, 1.0, 0.05)`

### Experiment 2 / NIR_cross_humidity (extracted from uploaded train.csv and test.csv)
- train humidities: `[0.0, 0.2, 0.4, 0.6, 0.8]`
- test humidities: `[0.1, 0.3, 0.5, 0.7, 0.9]`
- unique source split: train `50` / test `14`
- per-class unique source counts:
  - train: `{0: 11, 1: 11, 2: 19, 3: 9}`
  - test: `{0: 3, 1: 3, 2: 5, 3: 3}`

### Experiment 5.6 (repeated grouped source-level holdout)
- outer split seeds: `[42, 52, 62, 72, 82, 92, 102, 112, 122, 132]`
- fixed test quota per class: calcite=3, azurite=3, goethite=5, malachite=3
- train humidities: `[0.0, 0.2, 0.4, 0.6, 0.8]`
- test humidities: `[0.1, 0.3, 0.5, 0.7, 0.9]`
- inner grouped CV folds: `4`
- training seeds per split: `[42, 52, 62, 72, 82]`

### Experiment 5.10 (alternative non-overlapping humidity splits)
- `split_A_current`
  - train: `[0.10, 0.30, 0.50, 0.70, 0.90]`
  - test: `[0.00, 0.20, 0.40, 0.60, 0.80]`
- `split_B_shifted`
  - train: `[0.00, 0.18, 0.36, 0.54, 0.72]`
  - test: `[0.09, 0.27, 0.45, 0.63, 0.81]`
- `split_C_shifted`
  - train: `[0.05, 0.23, 0.41, 0.59, 0.77]`
  - test: `[0.14, 0.32, 0.50, 0.68, 0.86]`

## Notes

- This starter repository intentionally excludes checkpoints, pretrained weights, large cached benchmark tables, and raw mirrored USGS source files.
- `source_split_main.json` is no longer a placeholder: it has been filled from the uploaded `NIR_cross_humidity` train/test CSV files.
- `humidity_split_cross.json` now reflects the uploaded dataset values directly.

## Publication-ready path cleanup

- `source_split_main.json` now uses repository-relative paths under `external_data/usgs_splib07/` instead of machine-specific absolute paths.
