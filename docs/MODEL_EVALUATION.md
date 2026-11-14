# Model Evaluation — Milestone 5

See README Milestone 5 for execution, cohort, and formula definitions. This note supplies the calculation checklist and limitations expected in the final technical report.

**Candidate ranking protocol:** Training-only feature and model artifacts → all known catalog items minus each user's training history → top 5 / top 10 scores → held-out relevance (rating ≥ 4). Test ratings are read only by evaluation. Temporal cutoff and model/train fingerprint are checked.

**Cohort:** Evaluate users with ≥1 unseen relevant held-out item. Record how many held-out rows qualify and how many cannot be ranked; avoid reporting a ranking score for users with no relevant holdout. The same cohort is shared across both models.

**Precision@K** = hits / K. **Recall@K** = hits / relevant holdout count. **DCG@K** = Σ relevance(rank)/log₂(rank+1). **NDCG@K** = DCG/IDCG. **Coverage@K** = union of recommended items / catalog size. **Diversity@K** = mean pairwise Jaccard distance of genre sets among recommendations; metric is 0 when list length <2.

**Audit artifacts:** `recommendation_audit.csv` lists all recommended pairs and their binary relevance; `model_comparison.csv` contains one metric per model per K; `model_comparison.png` plots those exact CSV values. `evaluation_report.json` stores input hashes, cohort and protocol metadata, and MLflow logging status. Review all before screenshots.

**No RMSE/MAE:** The collaborative SVD algorithm reconstructs scores that are not calibrated to explicit 1–5 rating estimates; therefore error on predicted star ratings would be misleading. If a calibrated explicit-rating model is added later, compute RMSE/MAE on held-out ratings.

**Model-selection caution:** The test holdout is used for final evaluation; repeated tuning against it would make it a validation set. For rigorous comparisons, tune on an earlier train/validation chronology and reserve test for the final report.
