"""
Core trust model: weighted aggregation of 6 dimensions.
"""
import numpy as np


# Default equal weights
DEFAULT_WEIGHTS = {
    "transparency": 1/6,
    "fairness": 1/6,
    "privacy": 1/6,
    "robustness": 1/6,
    "explainability": 1/6,
    "accountability": 1/6,
}


def normalize_minmax(values):
    """Min-max normalization to [0, 1]."""
    arr = np.array(values, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-10:
        return np.full_like(arr, 0.5)
    return (arr - vmin) / (vmax - vmin)


def normalize_inverse(values):
    """Inverse normalization for risk metrics (lower is better)."""
    return 1 - normalize_minmax(values)


def compute_trust(metrics, weights=None):
    """
    Compute composite trust score T = sum(w_i * M_i).
    
    metrics: dict with keys MEI, BS, ARS, PS, GS
    weights: optional dict override
    """
    w = weights or DEFAULT_WEIGHTS.copy()
    
    # Map metric names to trust dimensions
    dimension_map = {
        "transparency": metrics.get("GS", 0.5),
        "fairness": metrics.get("BS", 0.5),
        "privacy": metrics.get("PS", 0.5),
        "robustness": metrics.get("ARS", 0.5),
        "explainability": metrics.get("MEI", 0.5),
        "accountability": metrics.get("GS", 0.5),
    }
    
    total = 0.0
    for dim, score in dimension_map.items():
        total += w.get(dim, 1/6) * score
    
    return round(total, 4)


def apply_sector_overlay(base_weights, sector):
    """Apply sector-specific weight adjustments (±0.05)."""
    w = base_weights.copy()
    if sector == "bfsi":
        w["transparency"] += 0.05
        w["accountability"] += 0.05
        w["fairness"] -= 0.03
        w["privacy"] -= 0.02
    elif sector == "healthcare":
        w["fairness"] += 0.05
        w["explainability"] += 0.05
        w["robustness"] -= 0.03
        w["accountability"] -= 0.02
    elif sector == "energy":
        w["robustness"] += 0.07
        w["privacy"] += 0.03
        w["transparency"] -= 0.05
        w["explainability"] -= 0.05
    
    # Renormalize to sum to 1
    total = sum(w.values())
    return {k: round(v / total, 4) for k, v in w.items()}


def trust_level(T):
    """Classify trust score into level."""
    if T > 0.8:
        return "High Trust"
    elif T > 0.5:
        return "Moderate Trust"
    else:
        return "Low Trust"
