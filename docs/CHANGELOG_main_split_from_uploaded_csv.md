# Main split update from uploaded NIR_cross_humidity CSV files

This update fills the main source split and main cross-humidity split using the uploaded files:

- `/mnt/data/train.csv`
- `/mnt/data/test.csv`

## Extracted unique source counts

- train unique sources: 50
- test unique sources: 14

Train per-class unique source counts:
{0: 11, 1: 11, 2: 19, 3: 9}

Test per-class unique source counts:
{0: 3, 1: 3, 2: 5, 3: 3}

## Extracted humidity levels

- train humidity levels: [0.0, 0.2, 0.4, 0.6, 0.8]
- test humidity levels: [0.1, 0.3, 0.5, 0.7, 0.9]

The file `source_split_main.json` is now populated with actual `source_file` entries extracted from the uploaded dataset.
