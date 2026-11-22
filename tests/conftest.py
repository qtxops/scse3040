import numpy as np
import pandas as pd
import pytest

from mp135.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def example_rows():
    index = np.arange(40)
    return pd.DataFrame({
        "record_id": [f"test-{i}" for i in index],
        "date": pd.date_range("2026-01-01", periods=40).strftime("%Y-%m-%d"),
        "hour": 6 + index % 18,
        "nearby_offices": 10 + index % 20,
        "temp_c": 20 + index % 10,
        "rain_mm": index % 30,
        "feeder_buses": 1 + index % 15,
        "ticket_price_rs": 10 + index % 30,
        "day_type": np.where(index % 2, "weekday", "weekend"),
        "line": np.where(index % 2, "Red Line", "Blue Line"),
        "entries_per_hour": 1000 + index * 25,
        "gate_count_end_of_hour": 1000 + index * 25 + 1,
    })


@pytest.fixture
def request_body():
    return {
        "hour": 9, "nearby_offices": 25, "temp_c": 30.0, "rain_mm": 5.0,
        "feeder_buses": 12, "ticket_price_rs": 30.0,
        "day_type": "weekday", "line": "Red Line",
    }
