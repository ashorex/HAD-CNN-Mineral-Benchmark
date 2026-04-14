# CHANGELOG: Section 5.6 repeated-holdout update

This update incorporates the uploaded Section 5.6 code files.

## What was added
- `benchmark/splits/repeated_holdout_scheme.json` is now populated from code-derived settings rather than left as a generic template.
- `benchmark/params/repeated_holdout_from_code.yaml` was added.

## Code-derived repeated-holdout settings
- outer split seeds: 42, 52, 62, 72, 82, 92, 102, 112, 122, 132
- test quota per class: 3 / 3 / 5 / 3
- train humidity: 0.0, 0.2, 0.4, 0.6, 0.8
- test humidity: 0.1, 0.3, 0.5, 0.7, 0.9
- inner grouped CV folds: 4
- per-split training seeds: 42, 52, 62, 72, 82
- epochs: 60

## Important note
The code-derived repeated-holdout humidity split is not identical to the Experiment 2 split described in the manuscript body.
