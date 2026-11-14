# Milestone 3 checklist and screenshot plan

- [x] User/item aggregates, preferred genre and recency from training-only data
- [x] Reusable candidate feature lookup, known-entity cold-start defaults
- [x] Content-addressed immutable feature snapshots and source/artifact checksums
- [x] Catalog with types, definitions, sources, null handling and temporal contract
- [x] Batch/prediction consistency demonstration
- [x] Automated tests for idempotence, tampering, lineage, leakage, and API behavior
- [ ] Execute with the full official 100K dataset on your Mac and capture genuine screenshots

## Evidence to capture on macOS

1. Terminal: `python -m scripts.pipeline full3 --require-clean` showing `FEATURES OK` and `FEATURE CONSISTENCY: True` after previous pipeline stages.
2. Terminal: `cat data/features/latest.json` and `cat reports/features/feature_report.json` showing actual feature version, cutoff and source hash.
3. File listing: `find data/features/snapshots -maxdepth 2 -type f | sort` showing user/item/seen tables and manifest.
4. CSV preview: `head -n 6 data/features/snapshots/*/user_features.csv` and corresponding item file (one active snapshot recommended).
5. Terminal: `python -m pytest -v` showing passing tests.
6. Include `docs/FEATURE_CATALOG.md` and feature design in final report.

Do not fabricate screenshots. If `full3` encounters network problems, download the official source using the prior tested instructions; for an already verified Milestone 2 directory, `python -m scripts.pipeline milestone3` reads the prepared files and does not download.
