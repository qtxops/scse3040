import numpy as np
import pandas as pd
import pytest

from mp135.monitoring import chronological_split, population_stability_index, promotion_decision


def test_psi_identical_samples_and_shift_with_out_of_range_values():
    reference = np.arange(100, dtype=float)
    assert population_stability_index(reference, reference) == pytest.approx(0)
    assert population_stability_index(reference, reference + 1000) > 1
    assert np.isfinite(population_stability_index([1, 1, np.nan], [1, 2, np.nan]))


@pytest.mark.parametrize("v1,v2,expected", [
    (100, 85, "promote"), (100, 85.01, "keep"), (100, 70, "promote"),
    (100, 110, "keep"), (0, 0, "keep"),
])
def test_exact_promotion_threshold(v1, v2, expected):
    assert promotion_decision(v1, v2) == expected


def test_later_split_is_chronological_and_disjoint():
    data = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=300).strftime("%Y-%m-%d"),
        "record_id": np.arange(300),
    }).sample(frac=1, random_state=42)
    update, check = chronological_split(data, 0.7)
    assert len(update) == 210
    assert len(check) == 90
    assert update["date"].max() <= check["date"].min()
    assert set(update["record_id"]).isdisjoint(check["record_id"])
