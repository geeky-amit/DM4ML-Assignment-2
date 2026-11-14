# DM4ML Assignment 2 — End-to-End Movie Recommendation Data Pipeline

**Individual assignment | MovieLens 100K | Python / Pandas / SciPy / MLflow / Prefect**

This is the full implementation covering **ingestion → validation → preparation → transformation → feature management → model training → model evaluation → orchestration**. The initial repository boilerplate has been updated for the final milestone. Source and automated tests are provided; locally generated data, model artifacts, MLflow databases and Prefect state must be regenerated on your machine.

**Submission status:** The college submission PDF is a **draft until** a genuine screenshot of the Completed Prefect flow and the actual public GitHub URL are supplied. Never submit example placeholders as evidence.

## Verified MovieLens execution (user-supplied macOS logs, 14 Nov 2026)

| Item | Verified result |
|---|---:|
| Original interactions / users / movies | 100,000 / 943 / 1,682 |
| Rejected records | 0 |
| Chronological train / test | 79,999 / 20,001 |
| CSR sparse matrix shape and nonzeros | 943 × 1,682; 79,999 |
| Feature snapshot | `v1_0_0_1b3a15ee2f84b168` |
| Evaluation cohort (rating >= 4) | 290 eligible users |
| Prefect flow (twice) | 8/8 stages Completed |
| Repeatability | `matched_previous_run` |
| Latest verified pytest screenshot | 48 passed (Milestone 5); Milestone 6's 56-test result was obtained in packaging environment, not yet screenshot-verified on the Mac |

See [`reports/submission_evidence/verified_execution_summary.json`](reports/submission_evidence/verified_execution_summary.json) and [`docs/RUBRIC_AUDIT.md`](docs/RUBRIC_AUDIT.md). These are a **transcription of user-provided output**, not re-execution within this final packaging environment.

## Environment setup (macOS)

Python **3.11** is a conservative recommendation for dependency compatibility. User's prior Mac execution was under Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -v
```

## Run the complete pipeline with Prefect

In terminal A, start the local Prefect UI:

```bash
prefect server start --host 127.0.0.1
```

In terminal B:

```bash
source .venv/bin/activate
export PREFECT_API_URL=http://127.0.0.1:4200/api
python -m scripts.pipeline milestone6 --tracking-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:4200 and inspect **dm4ml-recommendation-end-to-end**. Repeat the command; check that `reports/orchestration/latest_run.json` reports `reproducibility.status = matched_previous_run`. Verify the final metrics in `reports/evaluation/model_comparison.csv`. Use `python -m scripts.pipeline full5 --require-clean --tracking-uri sqlite:///mlflow.db` for a non-Prefect full run.

Model recommendations:

```bash
python -m scripts.pipeline recommend --user-id 1 --k 5
```

MLflow UI (matching the configured SQLite URI):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Browse http://127.0.0.1:5000 and verify both model runs, parameters and logged metrics. `--tracking-uri sqlite:///mlflow.db` is required if the default code/installed MLflow is configured to use the deprecated file store.

## Important documentation

- [`docs/architecture.png`](docs/architecture.png) — complete eight-stage pipeline diagram
- [`docs/FEATURE_CATALOG.md`](docs/FEATURE_CATALOG.md) — feature catalog, definitions, cold-start and temporal rules
- [`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md) — evaluation formulas and cohort contract
- [`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md) — dependencies, retries and reproducibility
- [`docs/EVIDENCE_CHECKLIST.md`](docs/EVIDENCE_CHECKLIST.md) — genuine screenshot capture checklist
- [`docs/RUBRIC_AUDIT.md`](docs/RUBRIC_AUDIT.md) — grading-requirement mapping and outstanding items
- [`submission/Technical_Report_3pages.pdf`](submission/Technical_Report_3pages.pdf) — 3-page final technical report
- [`submission/Execution_Evidence_DRAFT.pdf`](submission/Execution_Evidence_DRAFT.pdf) — genuine available screenshots plus missing-evidence notices

## Version control and data ethics

Do **not** commit downloaded MovieLens source, `.venv/`, model binaries, `mlflow.db`, `mlruns/`, `data/features/`, personal local paths or raw Prefect runtime state. `.gitignore` excludes generated artifacts. For public submission, attach concise redacted validation/evaluation/reproducibility **summaries** or regenerated screenshots, not the raw data. The original dataset is obtained from [GroupLens](https://grouplens.org/datasets/movielens/100k/).

## Evaluation protocol and limitations

Use the global chronological 80/20 split; retain timestamp ties. Only training interactions enter model fitting and feature snapshots. Evaluate previously unseen movies with held-out rating >= 4 as relevant. Precision, Recall, NDCG, Coverage and genre-based Jaccard Diversity are reported at K=5 and K=10. SVD scores are **not calibrated rating predictions**, so RMSE/MAE are not reported. Only 290 users meet cohort rules; 192 SVD-user recommendations use the popularity fallback. This limits generalization claims and should be transparently disclosed.

## Submission checklist

1. Publish your own public GitHub repository (there is **no** GitHub URL in these packaged artifacts).
2. Capture **genuine execution screenshots for every required pipeline stage** (the Prefect flow/task view supports stage-state verification; add stage terminal/report views). Include the **real Prefect Completed flow screenshot**, showing all task states. Optionally capture MLflow UI.
3. Place missing screenshots in `submission/evidence_assets/` and rebuild/re-export the PDF in a document editor, or include them alongside the current evidence appendix.
4. Replace the clearly marked GitHub placeholder in the submitted PDF, verify the link works in an incognito browser, and confirm the PDF includes all execution-stage evidence.
5. Submit the required PDF per the college announcement and GitHub URL.
# DM4ML-Assignment-2
