"""Find the seven specified data issues and clean training records only.

The clean later file supplies reference units and category spellings, never
targets for fitting v1. Original CSVs are read only.
"""

import re

import numpy as np
import pandas as pd


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def find_stuck_runs(frame, columns, minimum_days=7):
    """A stuck run has one identical reading on every row of consecutive days."""
    results = []
    dated = frame.assign(date=pd.to_datetime(frame["date"]))
    for column in columns:
        daily = dated.groupby("date")[column].agg(["nunique", "first"])
        start = previous = value = None
        days = 0

        def finish():
            if days >= minimum_days:
                results.append({
                    "column": column,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": previous.strftime("%Y-%m-%d"),
                    "value": float(value),
                    "days": days,
                })

        for date, row in daily.iterrows():
            same = (
                row["nunique"] == 1 and days > 0 and row["first"] == value
                and (date - previous).days == 1
            )
            if same:
                days += 1
            else:
                finish()
                days = 1 if row["nunique"] == 1 else 0
                start, value = date, row["first"]
            previous = date
        finish()
    return sorted(results, key=lambda item: item["days"], reverse=True)


def target_typo_mask(series):
    """Find a small upper cluster separated by a >=5-fold target gap."""
    ordered = np.sort(series.dropna().to_numpy(dtype=float))
    if len(ordered) < 2:
        return pd.Series(False, index=series.index)
    ratios = ordered[1:] / np.maximum(ordered[:-1], 1)
    for index in np.argsort(ratios)[::-1]:
        upper_count = len(ordered) - index - 1
        if ratios[index] >= 5 and upper_count <= max(1, len(ordered) * 0.05):
            return series >= ordered[index + 1]
    return pd.Series(False, index=series.index)


def audit_data(train, later, holdout, config):
    numeric = config["features"]["numeric"]
    categories = config["features"]["categorical"]
    target, group = config["target"], config["group_column"]
    issues = {
        "leaking_column": {"present": False, "column": None},
        "unit_mixup": {"present": False, "column": None, "group_value": None, "factor": None},
        "sentinel_value": {"present": False, "column": None, "value": None, "count": None},
        "duplicate_rows": {"present": False, "count": None},
        "stuck_sensor": {"present": False, "column": None, "start_date": None, "end_date": None},
        "target_typos": {"present": False, "count": None},
        "spelling_variants": {"present": False, "column": None, "true_levels": None},
    }
    evidence = {"rows": len(train), "correlations": {}, "unit_comparisons": []}

    # A post-event column is absent at inference and almost duplicates the target.
    for column in train.select_dtypes(include="number").columns:
        if column != target and column not in holdout.columns:
            correlation = float(train[column].corr(train[target]))
            evidence["correlations"][column] = correlation
            if abs(correlation) >= 0.98:
                issues["leaking_column"] = {"present": True, "column": column}

    # Match an exact conventional conversion, rather than reporting a rough ratio.
    factors = np.array([10, 12, 60, 100, 1000], dtype=float)
    for column in numeric:
        for level, rows in train.groupby(group):
            reference = later.loc[later[group] == level, column]
            if reference.empty or reference.median() <= 0:
                continue
            ratio = float(rows[column].median() / reference.median())
            evidence["unit_comparisons"].append({"column": column, "group": level, "median_ratio": ratio})
            factor = float(factors[np.argmin(np.abs(factors - ratio))])
            if ratio > 5 and abs(ratio / factor - 1) < 0.25:
                issues["unit_mixup"] = {
                    "present": True, "column": column, "group_value": level, "factor": int(factor),
                }

    for column in numeric:
        # All count, fare, hour and rainfall features must be nonnegative.
        invalid = train[column] < (-100 if column == "temp_c" else 0)
        if invalid.any():
            value = train.loc[invalid, column].mode().iloc[0]
            issues["sentinel_value"] = {
                "present": True, "column": column, "value": float(value),
                "count": int((train[column] == value).sum()),
            }

    duplicate_count = int(train.duplicated().sum())
    evidence["duplicate_count"] = duplicate_count
    if duplicate_count:
        issues["duplicate_rows"] = {"present": True, "count": duplicate_count}

    runs = find_stuck_runs(train, numeric)
    evidence["stuck_runs"] = runs
    if runs:
        issues["stuck_sensor"] = {"present": True, **{
            key: runs[0][key] for key in ["column", "start_date", "end_date"]
        }}

    typo_count = int(target_typo_mask(train[target]).sum())
    evidence["target_typo_count"] = typo_count
    values = np.sort(train[target].to_numpy(dtype=float))
    evidence["largest_target_gap_ratio"] = float(np.max(values[1:] / np.maximum(values[:-1], 1)))
    if typo_count:
        issues["target_typos"] = {"present": True, "count": typo_count}

    evidence["category_counts"] = {}
    for column in categories:
        reference = {normalized(value): value for value in later[column].dropna().unique()}
        cleaned = train[column].map(lambda value: reference.get(normalized(value), value))
        evidence["category_counts"][column] = {
            "raw": {str(k): int(v) for k, v in train[column].value_counts().items()},
            "clean": {str(k): int(v) for k, v in cleaned.value_counts().items()},
        }
        if not cleaned.equals(train[column]):
            issues["spelling_variants"] = {
                "present": True, "column": column, "true_levels": int(cleaned.nunique()),
            }
    return issues, evidence


def clean_training(train, issues, later):
    cleaned = train.drop_duplicates().copy()
    issue = issues["unit_mixup"]
    if issue["present"]:
        column = issue["column"]
        cleaned[column] = cleaned[column].astype(float)
        mask = cleaned["line"] == issue["group_value"]
        cleaned.loc[mask, column] /= issue["factor"]
    issue = issues["sentinel_value"]
    if issue["present"]:
        cleaned.loc[cleaned[issue["column"]] == issue["value"], issue["column"]] = np.nan
    issue = issues["stuck_sensor"]
    if issue["present"]:
        mask = cleaned["date"].between(issue["start_date"], issue["end_date"])
        cleaned.loc[mask, issue["column"]] = np.nan
    if issues["target_typos"]["present"]:
        mask = target_typo_mask(cleaned["entries_per_hour"])
        cleaned.loc[mask, "entries_per_hour"] /= 10
    issue = issues["spelling_variants"]
    if issue["present"]:
        column = issue["column"]
        reference = {normalized(value): value for value in later[column].dropna().unique()}
        cleaned[column] = cleaned[column].map(lambda value: reference.get(normalized(value), value))
    issue = issues["leaking_column"]
    if issue["present"]:
        cleaned = cleaned.drop(columns=issue["column"])
    return cleaned
