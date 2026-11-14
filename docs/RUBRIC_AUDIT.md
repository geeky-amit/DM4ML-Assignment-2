# Final rubric audit — DM4ML Assignment 2

Basis: original 3-page assignment, shipped Milestone 6 source files, user-provided Mac logs/screenshots. Tests validate defined behavior; final marking belongs to the college evaluator.

| Official category | Weight | Implemented / verified evidence | Outstanding |
|---|---:|---|---|
| Ingestion and validation | 20% | GroupLens SHA1, three source entities, type/missing/duplicate/rating/time/FK rules, quarantine, 0 rejected in original data | Include real validation execution screenshot in PDF |
| Preparation and transformation | 20% | Time-based 79,999/20,001 split, tie and leakage tests, aggregate tables, 943×1,682 CSR matrix | Include real preparation/transformation execution screenshot |
| Feature management | 15% | Train-only versioned snapshots, catalog, cold-start policy and candidate lookup | Include real feature/snapshot execution screenshot |
| Model training and evaluation | 25% | Popularity and SVD, MLflow SQLite run IDs, Precision/Recall/NDCG/Coverage/Diversity at 5 and 10, real CSV/PNG and user-supplied screenshots | MLflow GUI screenshot helpful; no calibrated star prediction hence no RMSE/MAE |
| Orchestration and reproducibility | 10% | Real Prefect macOS run logs 8/8 completed twice, repeat fingerprints matched | **Actual Prefect UI Completed screenshot not yet supplied** |
| Code quality, documentation, report | 10% | Modular Python, tests, README, diagram, catalog, 3-page PDF report | **Public GitHub URL must be added**; finish execution-evidence PDF |

## Known limitations

- 290 eligible users selected by at least one unseen relevant rating (>=4); this is not every user in the dataset.
- Personalized SVD fell back to popularity for 192/290 eligible users (66.2%). Interpret ranking gains carefully.
- Global chronological split is leakage-safe but causes cold-start imbalance.
- Model scores are not calibrated star-rating predictions; RMSE/MAE do not apply.
- MLflow tracking SQLite stores are local generated runtime artifacts, not distributed in this ZIP.
- Some automated tests were run via simulated Prefect interface in the packaging environment; genuine full-data Prefect flow run is independently shown by user terminal logs.
- Code uses version ranges in requirements.txt, not a fully locked dependency-resolution file.
- Generated evidence PDF includes only real uploaded images; Prefect run log has been transcribed as text because no Prefect dashboard screenshot is available.
