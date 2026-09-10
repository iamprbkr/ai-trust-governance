"""
Main evaluation script. Reproduces Tables 2-5 from the paper.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from metrics import compute_all_metrics
from trust_model import compute_trust, DEFAULT_WEIGHTS, apply_sector_overlay
from data.download import load_adult, load_german, load_compas


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def prepare_adult(df):
    """Prepare Adult Income dataset."""
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    y = (df["income"] > 0).astype(int)
    X = df.drop("income", axis=1)
    sensitive = [("sex", 1), ("race", 4)]
    return X.values, y.values, X.columns.tolist(), sensitive


def prepare_german(df):
    """Prepare German Credit dataset."""
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    y = (df["risk"] == 1).astype(int)
    X = df.drop("risk", axis=1)
    return X.values, y.values, X.columns.tolist(), []


def prepare_compas(df):
    """Prepare COMPAS dataset."""
    df = df.copy()
    cols_keep = ["age", "priors_count", "days_b_screening_arrest",
                 "c_offense_date", "v_decod_score", "decod_score",
                 "is_recid", "two_year_recid"]
    for col in cols_keep:
        if col not in df.columns:
            df[col] = 0
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    y = df["two_year_recid"].values
    X = df[cols_keep].values
    return X, y, cols_keep, [("age", 0)]


def run_experiment(name, X, y, sensitive, n_runs=10):
    """Run evaluation on a dataset."""
    results = []
    for seed in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
        model = RandomForestClassifier(n_estimators=100, random_state=seed)
        model.fit(X_train, y_train)
        
        metrics = compute_all_metrics(model, X_test, y_test, sensitive)
        
        # NIST-only: governance + robustness only
        nist_score = compute_trust(
            {"MEI": metrics["MEI"], "BS": 0.5, "ARS": metrics["ARS"],
             "PS": 0.5, "GS": metrics["GS"]},
            weights={"transparency": 0.25, "fairness": 0.10, "privacy": 0.10,
                     "robustness": 0.25, "explainability": 0.15, "accountability": 0.15}
        )
        
        # OWASP-only: security metrics only
        owasp_score = compute_trust(
            {"MEI": 0.5, "BS": 0.5, "ARS": metrics["ARS"],
             "PS": metrics["PS"], "GS": 0.5},
            weights={"transparency": 0.10, "fairness": 0.10, "privacy": 0.20,
                     "robustness": 0.30, "explainability": 0.10, "accountability": 0.20}
        )
        
        # Proposed: all dimensions, equal weights
        proposed_score = compute_trust(metrics)
        
        results.append({
            "run": seed + 1,
            "MEI": metrics["MEI"],
            "BS": metrics["BS"],
            "ARS": metrics["ARS"],
            "PS": metrics["PS"],
            "GS": metrics["GS"],
            "NIST_only": nist_score,
            "OWASP_only": owasp_score,
            "Proposed": proposed_score,
        })
    
    return pd.DataFrame(results)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    datasets = {
        "adult": (load_adult, prepare_adult),
        "german": (load_german, prepare_german),
        "compas": (load_compas, prepare_compas),
    }
    
    all_results = {}
    
    for name, (loader, preparer) in datasets.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {name.upper()}")
        print(f"{'='*50}")
        
        try:
            df = loader()
            X, y, cols, sensitive = preparer(df)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue
        
        results = run_experiment(name, X, y, sensitive)
        all_results[name] = results
        
        # Save
        out_path = os.path.join(RESULTS_DIR, f"{name}_results.csv")
        results.to_csv(out_path, index=False)
        
        # Print summary
        print(f"\n  Results ({len(results)} runs):")
        print(f"  NIST-only:   {results['NIST_only'].mean():.4f} ± {results['NIST_only'].std():.4f}")
        print(f"  OWASP-only:  {results['OWASP_only'].mean():.4f} ± {results['OWASP_only'].std():.4f}")
        print(f"  Proposed:    {results['Proposed'].mean():.4f} ± {results['Proposed'].std():.4f}")
    
    # Cross-dataset summary (Table 3)
    print(f"\n{'='*50}")
    print("Cross-Dataset Summary")
    print(f"{'='*50}")
    
    summary_rows = []
    for name, df in all_results.items():
        summary_rows.append({
            "Dataset": name.capitalize(),
            "NIST_only": f"{df['NIST_only'].mean():.2f}",
            "OWASP_only": f"{df['OWASP_only'].mean():.2f}",
            "Proposed": f"{df['Proposed'].mean():.2f} ± {df['Proposed'].std():.2f}",
        })
    
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(RESULTS_DIR, "cross_dataset_summary.csv"), index=False)
    
    # Statistical tests
    print(f"\n{'='*50}")
    print("Statistical Tests")
    print(f"{'='*50}")
    
    from scipy import stats
    
    proposed_all = np.concatenate([df["Proposed"].values for df in all_results.values()])
    nist_all = np.concatenate([df["NIST_only"].values for df in all_results.values()])
    owasp_all = np.concatenate([df["OWASP_only"].values for df in all_results.values()])
    
    t_nist, p_nist = stats.ttest_rel(proposed_all, nist_all)
    t_owasp, p_owasp = stats.ttest_rel(proposed_all, owasp_all)
    
    diff_nist = proposed_all - nist_all
    diff_owasp = proposed_all - owasp_all
    
    print(f"  Proposed vs NIST-only:  diff={diff_nist.mean():.4f}, p={p_nist:.4e}")
    print(f"  Proposed vs OWASP-only: diff={diff_owasp.mean():.4f}, p={p_owasp:.4e}")
    print(f"  95% CI (vs NIST):  [{diff_nist.mean() - 1.96*diff_nist.std():.4f}, {diff_nist.mean() + 1.96*diff_nist.std():.4f}]")
    print(f"  95% CI (vs OWASP): [{diff_owasp.mean() - 1.96*diff_owasp.std():.4f}, {diff_owasp.mean() + 1.96*diff_owasp.std():.4f}]")
    
    # Wilcoxon test
    w_nist, wp_nist = stats.wilcoxon(proposed_all, nist_all)
    w_owasp, wp_owasp = stats.wilcoxon(proposed_all, owasp_all)
    print(f"  Wilcoxon (vs NIST):  p={wp_nist:.4e}")
    print(f"  Wilcoxon (vs OWASP): p={wp_owasp:.4e}")
    
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
