# Submission evidence checklist (only genuine execution captures)

## Already supplied and included in evidence PDF

- [x] `EVALUATE OK` with 20,001 held-out interactions and both models
- [x] 48 passing Pytest tests in macOS screenshot (Milestone 5)
- [x] Evaluation report and macro metric definitions, 290 eligible users
- [x] Actual model comparison table at K=5 and K=10
- [x] Actual generated model-comparison chart
- [x] Real terminal log excerpt of two Prefect successful runs with repeated fingerprints (as **text**, not screenshot)

## Still to capture on Mac

- [ ] **Mandatory:** Open `http://127.0.0.1:4200`, flow `dm4ml-recommendation-end-to-end`, show `Completed` run and all eight task states; take a genuine screenshot
- [ ] **Mandatory:** Publish your public GitHub repo and paste its real URL into final PDF
- [ ] Strongly recommended: screenshot of MLflow UI at `http://127.0.0.1:5000` showing both model runs and metric values
- [ ] Recommended: screenshot of `python -m pytest -v` for Milestone 6 (56 passing tests, if reproduced on Mac)
- [ ] **Mandatory under the stated submission instructions:** Genuine execution screenshot coverage for **all eight pipeline stages**; the Prefect task view can support stage-state evidence, supplemented with terminal/report screenshots for ingestion, validation, split, features, training and evaluation

## Suggested screenshot commands

```bash
source .venv/bin/activate
export PREFECT_API_URL=http://127.0.0.1:4200/api
prefect server start --host 127.0.0.1              # separate terminal
python -m scripts.pipeline milestone6 --tracking-uri sqlite:///mlflow.db
python -m pytest -v
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000  # separate terminal
cat reports/orchestration/latest_run.json
cat reports/evaluation/evaluation_report.json
```

Screenshot quality: show app/terminal context, the executed command, finished state, date where available; crop out personal data and avoid altering success indicators. Do not pass this evidence draft off as a completed submission until marked gaps are resolved.
