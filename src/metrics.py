"""
Trust metrics computation: MEI, BS, ARS, PS, GS.
"""
import numpy as np
from sklearn.metrics import accuracy_score


def compute_mei(model, X, threshold=0.5):
    """
    Model Explainability Index.
    Fraction of predictions with SHAP explanations above threshold.
    Returns a proxy based on feature importance concentration.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # High concentration = more explainable
        normalized = importances / (importances.sum() + 1e-10)
        top_k = np.sort(normalized)[-3:].sum()
        return min(top_k, 1.0)
    # Fallback: use prediction confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidence = proba.max(axis=1).mean()
        return float(confidence)
    return 0.5


def compute_bs(model, X, sensitive_groups, privileged_group=1):
    """
    Bias Score (BS).
    BS = 1 - |P(A|G1) - P(A|G2)|
    where G1, G2 are demographic groups.
    """
    preds = model.predict(X)
    scores = []
    for group_col, priv_val in sensitive_groups:
        group_vals = X[group_col].values if hasattr(X, 'columns') else X[:, group_col]
        mask_priv = group_vals == priv_val
        mask_unpriv = group_vals != priv_val
        if mask_priv.sum() == 0 or mask_unpriv.sum() == 0:
            continue
        rate_priv = preds[mask_priv].mean()
        rate_unpriv = preds[mask_unpriv].mean()
        scores.append(1 - abs(rate_priv - rate_unpriv))
    return float(np.mean(scores)) if scores else 0.5


def compute_ars(model, X, y, perturbation_fraction=0.1, n_runs=5):
    """
    Adversarial Resilience Score (ARS).
    Accuracy under random feature perturbation.
    """
    rng = np.random.RandomState(42)
    n_features = X.shape[1]
    n_perturb = max(1, int(n_features * perturbation_fraction))
    original_acc = accuracy_score(y, model.predict(X))
    perturbed_accs = []
    for _ in range(n_runs):
        X_perturbed = X.copy() if hasattr(X, 'copy') else np.copy(X)
        perturb_cols = rng.choice(n_features, n_perturb, replace=False)
        for col in perturb_cols:
            col_vals = X_perturbed[:, col] if isinstance(X_perturbed, np.ndarray) else X_perturbed.iloc[:, col].values
            std = np.std(col_vals)
            noise = rng.normal(0, std * 0.3, size=col_vals.shape)
            if isinstance(X_perturbed, np.ndarray):
                X_perturbed[:, col] = col_vals + noise
            else:
                X_perturbed.iloc[:, col] = col_vals + noise
        perturbed_accs.append(accuracy_score(y, model.predict(X_perturbed)))
    avg_perturbed = np.mean(perturbed_accs)
    return float(avg_perturbed / (original_acc + 1e-10))


def compute_ps(X, model=None):
    """
    Privacy Score (PS).
    Proxy based on feature minimization — fewer features used = higher score.
    """
    if model is not None and hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Effective number of features (entropy-based)
        norm_imp = importances / (importances.sum() + 1e-10)
        entropy = -np.sum(norm_imp * np.log(norm_imp + 1e-10))
        max_entropy = np.log(len(importances))
        return float(1 - entropy / (max_entropy + 1e-10))
    return 0.5


def compute_gs(policy_traceability=0.7, audit_completeness=0.7):
    """
    Governance Score (GS).
    Policy traceability and audit completeness.
    """
    return (policy_traceability + audit_completeness) / 2


def compute_all_metrics(model, X, y, sensitive_groups=None):
    """Compute all five metrics."""
    mei = compute_mei(model, X)
    bs = compute_bs(model, X, sensitive_groups or [])
    ars = compute_ars(model, X, y)
    ps = compute_ps(X, model)
    gs = compute_gs()
    return {
        "MEI": round(mei, 4),
        "BS": round(bs, 4),
        "ARS": round(ars, 4),
        "PS": round(ps, 4),
        "GS": round(gs, 4),
    }
