# Data dictionary (Milestone 1)

## Source provenance
MovieLens 100K by GroupLens: https://grouplens.org/datasets/movielens/100k/ .
Download source: https://files.grouplens.org/datasets/movielens/ml-100k.zip .
The original files use legacy Latin-1 encoding for metadata. Ratings are tab-separated, and user and item data use `|` as delimiter. Source timestamps are Unix seconds UTC.

## Entities

| Table | Field | Type at source | Required | Validation |
| --- | --- | --- | --- | --- |
| users | user_id | positive integer text | yes | unique; FK target |
| users | age | integer text | yes | 1 through 120 |
| users | gender | category text | yes | M or F (source coding) |
| users | occupation | text | yes | nonempty |
| users | zip_code | text | yes | nonempty, preserve leading zeroes |
| items | item_id | positive integer text | yes | unique; FK target |
| items | title | text | yes | nonempty |
| items | release_date | dd-Mon-YYYY text | no | parse if nonempty |
| items | video_release_date | text | no | retained as supplied |
| items | imdb_url | text | no | retained as supplied |
| items | 19 genre columns | binary text | yes | each 0 or 1 |
| interactions | user_id | positive integer text | yes | exists in accepted users |
| interactions | item_id | positive integer text | yes | exists in accepted items |
| interactions | rating | integer text | yes | 1–5 |
| interactions | timestamp | Unix seconds UTC text | yes | integer, 1990-01-01 through configured `now` |

## Duplicate policy
A duplicate user or item entity ID is rejected after its first occurrence. A repeated `(user_id, item_id)` interaction is rejected after its first occurrence; the choice is documented and will be revisited if a source permits repeated ratings over time. **This is a cleaning assumption, not a guaranteed property of all recommender datasets.**

## Quality report
`reports/validation/validation_report.json` reports counts for every check and per-table status. Rejected rows in `data/errors/*_rejected.csv` include pipe-delimited error reasons. When a referenced entity fails validation, its interactions are also quarantined.

## Explicit scope boundary
Milestone 1 does not contain temporal splitting, features, training, evaluation, MLflow or orchestration. These are deliverables of later milestones.
