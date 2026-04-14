# Benchmark definition summary

This starter package consolidates benchmark settings for:

- Experiment 1 matched humidity
- Experiment 2 / uploaded `NIR_cross_humidity`
- Experiment 5.6 repeated grouped source-level holdout
- Experiment 5.10 alternative non-overlapping humidity splits

## Main extracted split

The uploaded `NIR_cross_humidity` files contain:
- train unique sources: 50
- test unique sources: 14
- train humidity levels: [0.0, 0.2, 0.4, 0.6, 0.8]
- test humidity levels: [0.1, 0.3, 0.5, 0.7, 0.9]

See:
- `benchmark/splits/source_split_main.json`
- `benchmark/splits/source_split_main_template.csv`
- `benchmark/splits/humidity_split_cross.json`
- `benchmark/splits/nir_cross_humidity_extracted_summary.json`
