# Milestone 3 — feature catalog (v1.0.0)

**Temporal contract:** all statistics below are computed exclusively from `data/processed/train_interactions.csv`, with `as_of_timestamp = max(train.timestamp)`. User and item identity/genre metadata comes from the prepared entity tables. `test_interactions.csv` is never read. Serving and post-cutoff evaluation must use the same immutable snapshot; do not train on historical train rating labels using snapshot aggregates that include those very labels.

| Feature | Entity / key | Type | Definition | Null / cold-start policy | Source |
|---|---|---|---|---|---|
| `user_activity_count` | user / `user_id` | int64 | Number of training ratings by user | 0 | Train ratings |
| `user_avg_rating` | user | float64 | Mean rating for user in training | missing/NaN | Train ratings |
| `user_last_timestamp` | user | float64 seconds | Unix timestamp of latest training rating | missing/NaN | Train ratings |
| `days_since_last_interaction` | user | float64 days | `(train_max_timestamp - user_last_timestamp) / 86400` | missing/NaN | Train ratings |
| `user_preferred_genre` | user | string | Training rating-weighted highest-scoring movie genre; tie broken by fixed genre order | empty string | Train ratings + item metadata |
| `item_popularity` | item / `item_id` | int64 | Number of training ratings received | 0 | Train ratings |
| `item_avg_rating` | item | float64 | Mean training rating for item | missing/NaN | Train ratings |
| `item_genres` | item | string | Pipe-delimited set of genre flags from MovieLens metadata | empty string | Item metadata |
| `seen_in_training` | user–item candidate | bool | Whether pair exists in training interactions | false for valid known pairs absent from train | Train interactions |
| `genre_affinity` | user–item candidate | int | 1 iff user's highest-preference genre belongs to item's genres | 0 for no preference/genre | User + item features |

`user_id` and `item_id` always denote entity keys rather than learned feature values. `item_genres` is based on original movie metadata rather than future ratings. No demographic attributes enter the feature layer by default.

## Snapshot versioning and reproducibility

Snapshots are stored under `data/features/snapshots/v1_0_0_<fingerprint>/`. Fingerprint is SHA256 of feature-layer semantic version plus SHA256 of the canonical prepared users, items, and training CSVs. Each snapshot contains `user_features.csv`, `item_features.csv`, `seen_train_pairs.csv`, optional matching Parquet files, and `manifest.json` with content hashes and cutoff. Existing snapshots are checked against their manifest and are not rewritten. Different prepared inputs or a new feature definition version create a different snapshot ID. `data/features/latest.json` is an index pointer; do not assume it alone establishes integrity.

## Candidate API

```python
import pandas as pd
from src.features.run import candidate_features

snapshot = 'data/features/snapshots/REPLACE_WITH_ACTUAL_SNAPSHOT_ID'
pairs = pd.DataFrame({'user_id': [1, 2], 'item_id': [10, 20]})
features = candidate_features(pairs, snapshot)
```

The candidate lookup uses shared, checked snapshot tables and preserves input order. It rejects unknown user/item IDs; it supports known-entity *zero-history* users/items via missing averages and zero counts. At recommendation time, filter `seen_in_training=True` candidates. The included consistency report compares two independent invocations of the exact same batch/prediction feature API; this verifies deterministic feature lookup (not production deployment or an online feature store).

**Future model integration:** choose a candidate pool without inspecting held-out labels. Use this snapshot only for predictions after its cutoff. For historical training examples, construct point-in-time features or a separate earlier training snapshot and ensure each example's label time follows the feature cutoff.
