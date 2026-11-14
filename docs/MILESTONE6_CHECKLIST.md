# Milestone 6 acceptance checklist

- [x] Prefect 3 flow code with eight stage tasks and explicit `wait_for` dependencies
- [x] Download retries; deterministic validation and downstream failure handling
- [x] Enforced zero-quarantine requirement and strict chronology
- [x] Existing MLflow SQLite `sqlite:///mlflow.db` integration preserved
- [x] Per-task receipt JSON plus completed/failed run report
- [x] Content-based repeatability fingerprint and baseline comparison
- [x] Automated tests for contract, fingerprints, simulated Prefect dependencies and failure propagation
- [x] Offline fixture-based end-to-end integration exercising real pipeline stages
- [x] Existing Milestones 1–5 CLI unchanged, additional Milestone 6 commands
- [ ] Full MovieLens 100K live Prefect run on macOS (student execution)
- [ ] Prefect UI completed flow and task evidence screenshots (student execution)
- [ ] MLflow UI evidence and final PDF/GitHub submission
