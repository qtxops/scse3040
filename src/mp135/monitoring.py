"""Training-reference PSI and the client's promotion decision."""

from numbers import Integral

import numpy as np
import pandas as pd


def population_stability_index(reference, current, bins=10):
    """Equal-width reference bins, open tails, missing values excluded.

    Open tails count readings outside the training range. Replace zero
    proportions by 0.0001 to keep the logarithm finite (P13 convention).
    """
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 2:
        raise ValueError("PSI bins must be an integer of at least 2")
    reference = pd.Series(reference).dropna().to_numpy(dtype=float)
    current = pd.Series(current).dropna().to_numpy(dtype=float)
    if not len(reference) or not len(current):
        raise ValueError("PSI needs nonempty reference and current samples")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("PSI samples must contain only finite values")
    low, high = float(reference.min()), float(reference.max())
    if low == high:
        edges = np.array([-np.inf, low - 0.5, high + 0.5, np.inf])
    else:
        edges = np.linspace(low, high, bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    expected = np.maximum(expected, 0.0001)
    actual = np.maximum(actual, 0.0001)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def chronological_split(frame, fraction):
    ordered = frame.sort_values("date", kind="stable").reset_index(drop=True)
    boundary = int(len(ordered) * fraction)
    if not 0 < boundary < len(ordered):
        raise ValueError("Both chronological partitions must contain rows")
    return ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()


def promotion_decision(mae_v1, mae_v2, minimum_improvement=0.15):
    return "promote" if mae_v1 > 0 and mae_v2 <= mae_v1 * (1 - minimum_improvement) else "keep"
