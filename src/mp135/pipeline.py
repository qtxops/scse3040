"""Run the assignment stages and write measured, reproducible findings."""

import hashlib
import json
import os
from copy import deepcopy

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error

from mp135.config import ROOT, load_config, project_path
from mp135.data import audit_data, clean_training
from mp135.modeling import build_model, predict
from mp135.monitoring import chronological_split, population_stability_index, promotion_decision


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def prepare_data(config):
    train = pd.read_csv(project_path(config, "train"))
    later = pd.read_csv(project_path(config, "later"))
    holdout = pd.read_csv(project_path(config, "holdout"))
    issues, evidence = audit_data(train, later, holdout, config)
    evidence["original_train_sha256"] = hashlib.sha256(
        project_path(config, "train").read_bytes(),
    ).hexdigest()
    cleaned = clean_training(train, issues, later)
    evidence["cleaned_rows"] = len(cleaned)
    evidence["cleaned_missing_counts"] = {
        column: int(cleaned[column].isna().sum())
        for column in config["features"]["numeric"]
    }
    path = project_path(config, "evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return cleaned, later, holdout, issues, evidence


def configure_tracking(config):
    # Resolve the required SQLite URI against the project even from another cwd.
    uri = config["tracking"]["uri"]
    if uri.startswith("sqlite:///"):
        database = uri.removeprefix("sqlite:///")
        uri = "sqlite:///" + str(ROOT / database)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    name = config["tracking"]["experiment_name"]
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(name)
    return client


def log_registered_model(model, features, config, register):
    options = {}
    if register:
        options["registered_model_name"] = config["tracking"]["registered_model_name"]
    return mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=features.head(3),
        signature=infer_signature(features, model.predict(features)),
        serialization_format="cloudpickle",
        pip_requirements=[
            line for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip()
        ],
        **options,
    )


def export_predictions(model, holdout, config):
    features = config["features"]["numeric"] + config["features"]["categorical"]
    result = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predict(model, holdout[features]),
    })
    if result["record_id"].duplicated().any() or not np.isfinite(result["prediction"]).all():
        raise ValueError("Holdout predictions must be finite and unique by record_id")
    path = project_path(config, "predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result


def run_pipeline(config):
    train, later, holdout, issues, evidence = prepare_data(config)
    client = configure_tracking(config)
    registered_name = config["tracking"]["registered_model_name"]
    existing = client.search_model_versions(f"name='{registered_name}'")
    if existing:
        raise RuntimeError(
            f"{registered_name} already has versions. Use 'python -m mp135 predict' "
            "to regenerate predictions, or choose a new registered_model_name in "
            "params.yaml to repeat training without overwriting recorded runs."
        )
    features = config["features"]["numeric"] + config["features"]["categorical"]
    target = config["target"]
    data_hash = evidence["original_train_sha256"]
    config_hash = hashlib.sha256((ROOT / "params.yaml").read_bytes()).hexdigest()
    fit, validation = chronological_split(train, 1 - config["training"]["validation_fraction"])
    trial_results = []
    for index, settings in enumerate(config["training"]["candidates"], start=1):
        with mlflow.start_run(run_name=f"trial_{index}") as run:
            mlflow.log_params({**settings, "data_sha256": data_hash, "params_sha256": config_hash})
            mlflow.set_tag("phase", "v1_selection")
            model = build_model(config, settings)
            model.fit(fit[features], fit[target])
            mae = float(mean_absolute_error(validation[target], predict(model, validation[features])))
            mlflow.log_metric("mae_validation", mae)
            mlflow.log_params({"fit_rows": len(fit), "validation_rows": len(validation)})
            trial_results.append({"settings": settings, "mae": mae, "run_id": run.info.run_id})
    best = min(trial_results, key=lambda result: result["mae"])
    v1 = build_model(config, best["settings"])
    v1.fit(train[features], train[target])
    with mlflow.start_run(run_name="v1") as run:
        mlflow.log_params({
            **best["settings"], "data_sha256": data_hash, "params_sha256": config_hash,
            "training_rows": len(train), "selection_run_id": best["run_id"],
        })
        mlflow.set_tag("phase", "v1")
        mlflow.log_metric("mae_validation", best["mae"])
        info = log_registered_model(v1, train[features], config, register=True)
        v1_run_id = run.info.run_id
        v1_version = str(info.registered_model_version)
    client.set_registered_model_alias(registered_name, "champion", v1_version)

    # Neither v1 selection nor fitting sees any later targets.
    monitoring = {}
    numeric_columns = train.select_dtypes(include="number").columns.intersection(later.columns)
    all_psi = {
        column: population_stability_index(train[column], later[column], config["monitoring"]["psi_bins"])
        for column in numeric_columns
    }
    monitoring["psi"] = {column: all_psi[column] for column in config["features"]["numeric"]}
    monitoring["most_drifted_feature"] = max(monitoring["psi"], key=monitoring["psi"].get)
    monitoring["mae_by_group"] = {
        str(level): float(mean_absolute_error(rows[target], predict(v1, rows[features])))
        for level, rows in later.groupby(config["group_column"])
    }
    monitoring["worst_served_group"] = max(monitoring["mae_by_group"], key=monitoring["mae_by_group"].get)
    update, check = chronological_split(later, config["monitoring"]["later_training_fraction"])
    expanded = pd.concat([train, update], ignore_index=True)
    v2 = build_model(config, deepcopy(best["settings"]))
    v2.fit(expanded[features], expanded[target])
    mae_v1 = float(mean_absolute_error(check[target], predict(v1, check[features])))
    mae_v2 = float(mean_absolute_error(check[target], predict(v2, check[features])))
    decision = promotion_decision(mae_v1, mae_v2, config["monitoring"]["minimum_improvement"])
    monitoring.update({"mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2, "decision": decision})
    with mlflow.start_run(run_name="v2"):
        mlflow.log_params({
            **best["settings"], "data_sha256": data_hash, "params_sha256": config_hash,
            "training_rows": len(expanded), "later_training_rows": len(update), "check_rows": len(check),
        })
        mlflow.set_tag("phase", "v2")
        mlflow.log_metrics({"mae_check": mae_v2, "mae_v1_on_check": mae_v1})
        mlflow.set_tag("promotion_decision", decision)
        info = log_registered_model(v2, expanded[features], config, register=True)
        v2_version = str(info.registered_model_version)
    champion_version = v2_version if decision == "promote" else v1_version
    client.set_registered_model_alias(registered_name, "champion", champion_version)

    champion = mlflow.sklearn.load_model(f"models:/{registered_name}@champion")
    model_path = project_path(config, "model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, model_path)
    predictions = export_predictions(champion, holdout, config)
    with project_path(config, "findings_template").open(encoding="utf-8") as handle:
        findings = yaml.safe_load(handle)
    findings.update({
        "data_issues": issues,
        "tracking": {
            "experiment_name": config["tracking"]["experiment_name"],
            "best_run_id": v1_run_id,
            "registered_model_name": registered_name,
            "champion_version": int(champion_version),
        },
        "monitoring": monitoring,
    })
    write_yaml(project_path(config, "findings"), findings)
    metrics = {
        "trials": trial_results, "psi_all_numeric_columns": all_psi,
        "later_training_rows": len(update), "later_check_rows": len(check),
        "later_training_date_range": [str(update["date"].min()), str(update["date"].max())],
        "later_check_date_range": [str(check["date"].min()), str(check["date"].max())],
        "relative_mae_improvement": (mae_v1 - mae_v2) / mae_v1 if mae_v1 else 0.0,
        "holdout_rows": len(predictions), "champion_version": int(champion_version),
        "serving_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"champion_version": champion_version, **monitoring, "holdout_rows": len(predictions)}, indent=2))


def regenerate_predictions(config):
    configure_tracking(config)
    name = config["tracking"]["registered_model_name"]
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")
    holdout = pd.read_csv(project_path(config, "holdout"))
    result = export_predictions(model, holdout, config)
    print(f"Wrote {len(result)} predictions from {name}@champion")
