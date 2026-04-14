# External data preparation

This repository uses public source spectra from the USGS Spectral Library (Version 7).

## Expected directory layout

```text
external_data/usgs_splib07/
└── ASCIIData/
    ├── ASCIIdata_splib07a/
    │   ├── ChapterM_Minerals/
    │   └── splib07a_Wavelengths_AVIRIS_1996_0.37-2.5_microns.txt
    └── ASCIIdata_splib07b/
        ├── ChapterM_Minerals/
        └── splib07b_Wavelengths_AVIRIS_1996_interp_to_2203ch.txt
```

## Path convention used in split files

All entries in `benchmark/splits/source_split_main.json` are normalized to repository-relative paths under:

`external_data/usgs_splib07/`

This avoids exposing machine-specific absolute paths such as `/autodl-fs/...`.

## Notes

- The repository does not redistribute the USGS raw source spectra.
- Benchmark tables are regenerated locally from the public source spectra and the released scripts.
