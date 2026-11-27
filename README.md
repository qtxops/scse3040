# G031 — MP135: Metro Station Crowd (Patna)

SCSE3040 MLOps mini project. Predict `entries_per_hour` with a regression
pipeline and report mean absolute error (MAE), in people.

| Enrolment number | Member |
|---|---|
| S24CSEU2633 | Sejal Shandilya |
| S24CSEU0991 | Shivam Sharma |
| S24CSEU2623 | Chirag Jain |
| S24CSEU2581 | Vaibhav Malik |

Faculty: Amit Soni. Batch: 70.

## Setup and tests

Use Python 3.12. Run these commands from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
pytest -q
```

Alternatively, skip the editable install and prefix pipeline/API commands
with `PYTHONPATH=src`. Tests configure this path automatically.

Tests create synthetic data and inject a test model into FastAPI. They do
not need the original training CSVs or a running MLflow server.

To verify the committed predictions against the exported model, check the
MLflow runs and champion alias, and validate the recorded promotion decision:

```sh
PYTHONPATH=src python scripts/check_artifacts.py
```

This check reads the database without changing it and needs only committed
files. It does not need the raw training CSVs or local MLflow model artifacts.

## Data and cleaning

Copy the three original CSVs from the assignment zip to `data/raw/`:
`train.csv`, `later.csv`, and `holdout_features.csv`. Never resave the raw
files through Excel. Training and later CSVs are tracked with DVC; their
`.dvc` metadata is committed and the CSVs/cache are ignored. The holdout
features are committed. No shared DVC remote is configured; a fresh clone
needs the two original CSVs from the assignment zip to rerun the audit.

Original training SHA-256:
`8674dc956f0becd5afb5c8d7fa09f3ffcdce6a57824df9d8196dd995747a9580`.

```sh
python -m mp135 audit
dvc status
```

`src/mp135/data.py` checks all seven issues and writes the evidence to
`reports/data_audit.json`. The measured findings are:

| Issue | Result and treatment |
|---|---|
| Leakage | Exclude `gate_count_end_of_hour`: recorded after the hour, almost identical to the target, absent from holdout. |
| Unit mix-up | Red Line `ticket_price_rs` values are 100 times too large; divide that group's training fares by 100. |
| Sentinel | 69 `nearby_offices` readings equal `-1`; replace with missing values. |
| Duplicate rows | Absent; zero extra copies. |
| Stuck sensor | `feeder_buses` is 13 throughout 29 March–19 April 2026; replace those 123 readings with missing values. |
| Target typos | Absent; no small upper cluster separated by a fivefold or larger target gap. |
| Spelling variants | Absent; category spellings already match the clean reference. |

Missing features are imputed using medians learned inside the model's
training partition. Numeric features are scaled, categories are one-hot
encoded, and all unlisted columns are discarded. IDs, dates, the target,
and the post-event gate counter never enter the estimator. Later data is
already clean: the training-only unit and sensor corrections are not
applied to live requests or later/holdout rows.

## Training, registry and monitoring

Paths, features, candidate settings and thresholds are in `params.yaml`,
read using `yaml.safe_load`.

```sh
python -m mp135 run
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The completed database is committed. Training tries Ridge with alpha 1
and 20, and histogram gradient boosting. Selection uses the last 20% of
cleaned training records by date, with preprocessing fitted on the first
80% only. Ridge with alpha 20 wins (validation MAE 156.23), and is refitted
on all 1,000 cleaned training records to produce registered version 1.
Every trial, v1 and v2 run records `data_sha256` and an MAE metric.

The clean later file is stably sorted by date. V2 fits the same selected
configuration on the 1,000 training rows plus the first 210 later rows.
Both versions are evaluated on the remaining 90 rows. An exact 70/30 row
split can place records dated 14 August on both sides; no check row is
used for fitting.

| Measurement | Result |
|---|---:|
| V1 MAE on the 90 check rows | 380.42 |
| V2 MAE on the same rows | 314.17 |
| Relative MAE improvement | 17.41% |
| Promotion threshold | At least 15% |
| Decision | Promote version 2 |
| Most drifted input | `rain_mm`, PSI 3.62256 |
| Worst-served line under v1 on all later rows | Blue Line, MAE 569.78 |

PSI uses 10 equal-width bins derived from the cleaned training range,
open-ended outer bins, zero proportions floored to 0.0001, and excludes
missing readings. All numeric-column PSI results, including the target,
are in `reports/metrics.json`; the six required input-feature values are
in `reports/findings.yaml`.

Registered model: `mp135-crowd`. Alias: `champion` → version **2**.
The pipeline loads that alias before exporting `models/model.joblib` and
the 250 predictions in `predictions/holdout_predictions.csv`. Negative
count predictions are clipped to zero consistently for evaluation, batch
predictions and API responses. Holdout labels are unavailable, so no
holdout MAE is claimed.

MLflow model artifacts remain in the local ignored `mlartifacts/` folder;
the committed exported champion supports serving from a fresh clone.
Registry-based prediction regeneration needs those local MLflow artifacts:

```sh
python -m mp135 predict
```

Training refuses to overwrite an existing registered model's versions.
To repeat training, choose a new `registered_model_name` in `params.yaml`;
the old runs remain in the database. The new model's versions start at 1
and 2, and its champion/exported results replace the report outputs.

## API

```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"hour":9,"nearby_offices":25,"temp_c":30,"rain_mm":5,"feeder_buses":12,"ticket_price_rs":30,"day_type":"weekday","line":"Red Line"}'
```

`GET /health` returns `{"status":"ok"}`. `POST /predict` returns
`{"prediction": <number>}`. Its schema requires exactly the eight input
fields shown above. Pydantic rejects `rain_mm > 100` and
`ticket_price_rs < 5` with HTTP 422 before invoking the model. Extra
fields, missing fields, invalid categories and nonfinite values are
rejected. Each successful prediction produces a JSON log line on stdout
containing its timestamp, prediction and latency, without request details.
Nonfinite model outputs return HTTP 500 before negative predictions are
clipped to zero.

## Docker, CI and Kubernetes

```sh
docker build -t mp135-api:1.0 .
docker run --rm -p 8000:8000 mp135-api:1.0
```

The Dockerfile uses `python:3.12-slim` and installs dependencies before
copying code. GitHub Actions installs requirements and runs pytest on
push and pull requests, checks the committed artifacts, and builds and
starts the container in a separate job. The live smoke check covers health,
real-model predictions and both client input rules. Run the same check
against your running API with:

```sh
python scripts/smoke_api.py http://127.0.0.1:8000
```

Check the repository's Actions tab for the actual remote result.

Kubernetes files define Deployment `mp135-api`, image `mp135-api:1.0`,
three replicas, and Service `mp135-svc` on port 80 targeting 8000.
Readiness and liveness probe `/health`; their initial delays are 5 and
20 seconds. A cluster is not required for the assignment. To deploy to
a local cluster, first load the built image into that cluster, then run:

```sh
kubectl apply -f k8s/
```

## Submission

Submit the same public repository URL from every member's form before
29 November 2026, 11:59 PM. Each member must contribute from their own
GitHub account and describe their actual work. Forms, individual
contributions and the viva must be completed by the members themselves.
The submission form link is in the assignment email.
