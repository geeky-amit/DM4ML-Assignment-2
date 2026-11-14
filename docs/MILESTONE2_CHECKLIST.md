# Milestone 2 — evidence and assessment checklist

## Assignment requirements addressed

- [x] Consume output from automated validation (not raw or rejected rows)
- [x] Standardize user/movie/interactions schema and timestamps
- [x] Deterministic **time-based** train/test split
- [x] Strict temporal split across identical timestamps
- [x] Training-only user-activity summaries
- [x] Training-only item-popularity summaries
- [x] Sparse training-only user–item interaction matrix
- [x] User/movie matrix index maps
- [x] Train/test and aggregate exports in CSV + optional Parquet
- [x] Automated unit, integration and CLI tests
- [x] Data lineage checksums and JSON audit reports
- [x] Preserved Milestone 1 CLI behavior
- [ ] Full MovieLens 100K preparation/transformation execution on student macOS
- [ ] Screenshots saved from student execution

## Test execution

`python -m pytest -v`

Tests use small, deterministic synthetic MovieLens-formatted fixtures. They test chronological boundaries, timestamp tie handling, invalid ratio, cold start, corrupted validated data, sparse matrix values, training-only aggregation isolation, and a chained CLI invocation. Full-dataset confirmation should be run on the actual official MovieLens files.

## Boundaries and limitations

- Global time holdout is used, not user-wise leave-last-out, to enforce temporal ordering.
- Held-out users and items that were absent in training are **reported** but model cold-start fallback will be implemented in a later milestone.
- `user_avg_rating` and `item_avg_rating` are null for entities with no training observations; this is intentional.
- No recommender or model evaluation metric is claimed yet.
- Outputs are files; feature versioning / serving comes in Milestone 3.
