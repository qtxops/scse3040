"""Validate committed submission outputs without raw CSVs or MLflow artifacts."""

import hashlib
import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import yaml

from mp135.config import ROOT, load_config, project_path
from mp135.modeling import predict
from mp135.monitoring import promotion_decision


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    config = load_config()
    findings = yaml.safe_load(project_path(config, "findings").read_text())
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    evidence = json.loads(project_path(config, "evidence").read_text())
    holdout = pd.read_csv(project_path(config, "holdout"))
    saved = pd.read_csv(project_path(config, "predictions"))
    model_path = project_path(config, "model")

    require(list(saved.columns) == ["record_id", "prediction"], "Unexpected prediction columns")
    require(saved["record_id"].is_unique, "Duplicate prediction record IDs")
    require(saved["record_id"].tolist() == holdout["record_id"].tolist(), "Holdout record IDs do not match")
    require(np.isfinite(saved["prediction"]).all(), "Nonfinite saved predictions")
    require((saved["prediction"] >= 0).all(), "Negative saved predictions")
    require(len(saved) == metrics["holdout_rows"], "Reported holdout row count does not match")
    require(
        hashlib.sha256(model_path.read_bytes()).hexdigest() == metrics["serving_model_sha256"],
        "Exported model hash does not match the report",
    )
    features = config["features"]["numeric"] + config["features"]["categorical"]
    regenerated = predict(joblib.load(model_path), holdout[features])
    np.testing.assert_allclose(saved["prediction"], regenerated, rtol=1e-9, atol=1e-9)

    name = config["tracking"]["registered_model_name"]
    champion = findings["tracking"]["champion_version"]
    require(champion == metrics["champion_version"], "Champion versions disagree between reports")
    with sqlite3.connect((ROOT / "mlflow.db").as_uri() + "?mode=ro", uri=True) as database:
        require(database.execute("pragma integrity_check").fetchone()[0] == "ok", "Invalid MLflow database")
        versions = database.execute("select version from model_versions where name=?", (name,)).fetchall()
        require({1, 2}.issubset({row[0] for row in versions}), "Registered v1 or v2 is missing")
        alias = database.execute(
            "select version from registered_model_aliases where name=? and alias='champion'", (name,),
        ).fetchone()
        require(alias == (champion,), "MLflow champion alias does not match the report")
        runs = database.execute(
            "select run_uuid,status from runs where lifecycle_stage='active'",
        ).fetchall()
        require(len(runs) >= 3, "Fewer than three active MLflow runs")
        require(findings["tracking"]["best_run_id"] in {row[0] for row in runs}, "Reported best run is missing")
        for run_id, status in runs:
            require(status == "FINISHED", f"MLflow run {run_id} did not finish")
            data_hash = database.execute(
                "select value from params where run_uuid=? and key='data_sha256'", (run_id,),
            ).fetchone()
            require(data_hash == (evidence["original_train_sha256"],), f"Data hash differs for run {run_id}")
            mae_count = database.execute(
                "select count(*) from metrics where run_uuid=? and lower(key) like '%mae%'", (run_id,),
            ).fetchone()[0]
            require(mae_count > 0, f"MAE metric is missing for run {run_id}")

    monitoring = findings["monitoring"]
    decision = promotion_decision(
        monitoring["mae_v1_on_check"], monitoring["mae_v2_on_check"],
        config["monitoring"]["minimum_improvement"],
    )
    require(decision == monitoring["decision"], "Promotion decision does not follow the configured rule")
    require(champion == (2 if decision == "promote" else 1), "Champion disagrees with the promotion decision")
    print(json.dumps({"holdout_rows": len(saved), "champion_version": champion, "mlflow_runs": len(runs), "status": "ok"}))


if __name__ == "__main__":
    main()
