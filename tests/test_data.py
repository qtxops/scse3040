import numpy as np
import pandas as pd

from mp135.data import audit_data, clean_training, find_stuck_runs, target_typo_mask


def test_audit_detects_units_missing_leak_and_duplicates(example_rows, config):
    later = example_rows.copy()
    holdout = example_rows.drop(columns=["entries_per_hour", "gate_count_end_of_hour"])
    messy = example_rows.copy()
    messy.loc[messy["line"] == "Red Line", "ticket_price_rs"] *= 100
    messy.loc[0, "nearby_offices"] = -1
    messy = pd.concat([messy, messy.iloc[[1]]], ignore_index=True)

    issues, _ = audit_data(messy, later, holdout, config)
    assert issues["leaking_column"]["column"] == "gate_count_end_of_hour"
    assert issues["unit_mixup"] == {
        "present": True, "column": "ticket_price_rs", "group_value": "Red Line", "factor": 100,
    }
    assert issues["sentinel_value"]["count"] == 1
    assert issues["duplicate_rows"]["count"] == 1
    assert issues["target_typos"]["present"] is False
    assert issues["spelling_variants"]["present"] is False

    clean = clean_training(messy, issues, later)
    assert len(clean) == len(example_rows)
    assert "gate_count_end_of_hour" not in clean
    assert np.isnan(clean.loc[0, "nearby_offices"])
    np.testing.assert_allclose(clean["ticket_price_rs"], later["ticket_price_rs"])
    assert messy.loc[0, "nearby_offices"] == -1


def test_stuck_sensor_requires_consecutive_days():
    data = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=12).strftime("%Y-%m-%d"),
        "feeder_buses": [4] + [13] * 10 + [8],
    })
    runs = find_stuck_runs(data, ["feeder_buses"])
    assert len(runs) == 1
    assert runs[0]["start_date"] == "2026-03-02"
    assert runs[0]["end_date"] == "2026-03-11"
    # A gap between isolated constant readings must break the run.
    gapped = data.iloc[[1, 2, 3, 8, 9, 10]]
    assert find_stuck_runs(gapped, ["feeder_buses"], minimum_days=4) == []


def test_target_typo_detection_requires_separated_small_cluster():
    normal = pd.Series([100, 110, 120, 115, 130] * 10)
    assert not target_typo_mask(normal).any()
    messy = pd.concat([normal, pd.Series([1500])], ignore_index=True)
    assert target_typo_mask(messy).sum() == 1
    assert target_typo_mask(messy).iloc[-1]


def test_spelling_variants_use_reference_categories(example_rows, config):
    later = example_rows.copy()
    messy = example_rows.copy()
    messy.loc[1, "day_type"] = " WEEKDAY "
    holdout = messy.drop(columns=["entries_per_hour", "gate_count_end_of_hour"])
    issues, _ = audit_data(messy, later, holdout, config)
    assert issues["spelling_variants"] == {
        "present": True, "column": "day_type", "true_levels": 2,
    }
    assert clean_training(messy, issues, later).loc[1, "day_type"] == "weekday"
