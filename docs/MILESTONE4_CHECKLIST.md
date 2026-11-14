# Milestone 4 — Model Training: Rubric Evidence

- [x] Popularity baseline: training interaction counts, deterministic tie-break by item ID
- [x] Personalized collaborative filtering: truncated sparse SVD (ARPACK), fixed seed
- [x] Candidate policy: known items only, exclude training-seen movies, no test labels
- [x] Cold-start: no/unknown user history falls back to popularity
- [x] Feature snapshot and train matrix checksum checks
- [x] Persisted NumPy .npz models, checksum manifest and reloadable inference
- [x] MLflow integration for two runs and associated artifacts (live MLflow execution must be confirmed on macOS)
- [x] Configurable factor count, seed, tracking URI and CLI
- [x] Automated tests including deterministic fixtures and API-level MLflow mock
- [ ] Live MLflow tracking screenshot, Mac execution evidence
- [ ] Ranking metrics (Precision/Recall/NDCG/Coverage/Diversity) — Milestone 5

## Screenshot capture

1. `python -m scripts.pipeline full4 --require-clean` — source, validation, preparation, feature snapshot and `TRAIN OK`.
2. `python -m scripts.pipeline recommend --user-id 1 --k 5` — rankings from each model.
3. `cat reports/training/training_report.json` — training metadata, snapshot version, two MLflow run IDs.
4. `mlflow ui --backend-store-uri ./mlruns --port 5000` — capture experiment runs, parameters and artifacts.
5. `python -m pytest -v` — tests passing on macOS.

*The `mlflow ui` command starts a local server: keep it open in a separate terminal while using http://127.0.0.1:5000.*
