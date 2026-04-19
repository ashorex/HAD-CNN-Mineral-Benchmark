# Benchmark definitions

This folder stores benchmark split definitions and default parameter files extracted from the manuscript, aligned code, and uploaded dataset files.

## Included split files

- `source_split_main.json`: main source-file-level split extracted from uploaded `train.csv` and `test.csv`
- `source_split_main_template.csv`: flattened manifest of the same extracted split
- `source_split_train_extracted.csv`: train unique sources extracted from uploaded `train.csv`
- `source_split_test_extracted.csv`: test unique sources extracted from uploaded `test.csv`
- `humidity_split_matched.json`: Experiment 1 matched-humidity setting
- `humidity_split_cross.json`: Experiment 2 / `NIR_cross_humidity` split extracted from uploaded `train.csv` and `test.csv`
- `nir_cross_humidity_extracted_summary.json`: summary extracted from uploaded `train.csv` and `test.csv`
- `humidity_split_alt_A.json`: Experiment 5.10 preset `split_A_current`
- `humidity_split_alt_B.json`: Experiment 5.10 preset `split_B_shifted`
- `humidity_split_alt_C.json`: Experiment 5.10 preset `split_C_shifted`
- `repeated_holdout_scheme.json`: Experiment 5.6 repeated grouped source-level holdout settings
- `exp510_scheme_from_code.json`: direct summary of the uploaded Experiment 5.10 code settings

## Important note

The uploaded `NIR_cross_humidity` dataset uses:
- train humidities: [0.1, 0.3, 0.5, 0.7, 0.9]
- test humidities: [0.0, 0.2, 0.4, 0.6, 0.8]
