# Reproducibility notes

This starter package is designed to support a code release consistent with the manuscript's reproducibility claims.

Included:
- split definitions
- parameter files
- repository scaffolding
- documentation templates

Excluded on purpose:
- pretrained weights / checkpoints
- mirrored raw USGS data
- large cached benchmark tables
- full raw experimental outputs

## Main split and cross-humidity alignment

The main source-file split and main `NIR_cross_humidity` split have been aligned to the uploaded dataset files.

Extracted values:
- train unique sources: 50
- test unique sources: 14
- train humidity levels: [0.0, 0.2, 0.4, 0.6, 0.8]
- test humidity levels: [0.1, 0.3, 0.5, 0.7, 0.9]

## Remaining manual items

You may still want to normalize absolute source file paths in `source_split_main.json` before publishing the repository, but the split content itself has now been filled.
