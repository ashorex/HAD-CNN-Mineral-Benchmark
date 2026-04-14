# Path normalization update

This update converts the extracted `source_file` entries from uploaded CSV files into repository-relative paths.

Before:
- machine-specific absolute paths such as `/autodl-fs/data/THZ_Humidity/usgs_splib07/...`

After:
- repository-relative paths such as `external_data/usgs_splib07/ASCIIData/...`

This version is safer to publish on GitHub.
