# Milestone 4: Recommender training design

## Scope

The input is the exact training-only `user_item_ratings_train.npz` sparse matrix from Milestone 2 and the immutable Milestone 3 feature snapshot. The trainer checks the prepared training CSV hash against the feature snapshot and transformation report, checks the sparse-matrix hash, checks entity membership and counts, and verifies matrix values for every training rating. **The trainer does not open `test_interactions.csv`.**

## Model 1: popularity baseline

Score of item `i` = number of ratings received in the training period. The score does not use ratings from the holdout or any outside item metadata. For equal scores, recommend lower item IDs first. Exclude each user's previously rated movies.

## Model 2: collaborative filtering

Sparse truncated SVD of the user × item explicit-rating matrix (not a calibrated rating predictor). It factorizes training matrix `R ≈ (U Σ) Vᵀ`, using ARPACK with fixed seeded starting vector, `k = min(requested_factors, min(number_of_users, number_of_items)-1)`. User-item affinity is the dot product of stored user and item factors. This is a *ranking score*, not predicted stars, and should not be described as calibrated 1–5 ratings. It cannot estimate preferences for entirely new users; a popularity fallback is used. Candidates are all catalog items minus the user's training-rated movies, including items with no historical ratings. The factorization only sees the training matrix. Numerical singular-vector signs are canonicalized for stable serialization.

## Training-time feature management

The training step loads and verifies the shared immutable feature snapshot, then uses the training matrix for collaborative filtering. Snapshot user/item metadata is used for consistency checks and provenance; the models **do not claim to train directly on snapshot aggregates**, which themselves contain historical training ratings. They are suitable for post-cutoff candidate ranking. Snapshot ID and source SHA256 are recorded in model manifest and both MLflow runs.

## Model artifacts

`models/popularity.npz` and `models/collaborative_svd.npz` contain IDs, popularity counts, training-seen CSR indices, and (for collaborative filtering) SVD factors. `models/manifest.json` records model version, source checksums, parameters and output SHA256 checksums. Loader requires the same manifest and checks each model artifact SHA256. Files are read with `allow_pickle=False`.

## MLflow experiments

Normal model training requires MLflow 3. The CLI logs two named runs with the snapshot, train SHA256, seed, requested/actual rank, model kind, training dimensions and model artifacts. It uses a local `./mlruns` tracking store unless `--tracking-uri` is supplied. **Training diagnostics are not evaluation metrics**; all five mandatory ranking metrics will be implemented using a leak-safe holdout in Milestone 5.

`--no-mlflow` permits isolated, offline testing only. It intentionally disables MLflow compliance for that run; do not use this mode as the final MLflow submission evidence.

## Known limitations and safeguards

- SVD on uncentered explicit ratings can favor globally popular items; comparative results remain unknown until evaluation.
- No ranking metrics have been computed yet.
- Cold-start catalog items may obtain near-zero factor scores; a model score is not a calibrated star rating.
- Only users/items present in the validated full catalog are represented; previously unseen catalog IDs require an upstream ingestion update.
- A cold-start user gets popularity recommendations; no demographic inference is made.
- For identical code/data/config the ranking output is deterministic. Byte-for-byte artifact equality is exercised on the test fixture, but may vary across BLAS/SciPy builds.
- `mlflow` is an optional *test-time* import but required for normal runs; `--no-mlflow` makes an explicit exception.
