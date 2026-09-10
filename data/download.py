"""
Data download script for Cyber Trust by Design.
Fetches the 5 datasets used in the evaluation.
"""
import os
import urllib.request
import zipfile
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download_file(url, dest):
    """Download a file if it doesn't exist."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"Already exists: {dest}")


def load_adult():
    """UCI Adult Income dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    dest = os.path.join(DATA_DIR, "adult.csv")
    download_file(url, dest)
    cols = ["age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    df = pd.read_csv(dest, names=cols, na_values=" ?", skipinitialspace=True)
    df = df.dropna()
    return df


def load_german():
    """UCI German Credit dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    dest = os.path.join(DATA_DIR, "german.csv")
    download_file(url, dest)
    cols = ["checking-account", "duration", "credit-history", "purpose", "credit-amount",
            "savings-account", "employment", "installment-rate", "personal-status",
            "other-debtors", "residence-since", "property", "age", "other-installment",
            "housing", "existing-credits", "job", "dependents", "telephone", "foreign", "risk"]
    df = pd.read_csv(dest, names=cols)
    return df


def load_compas():
    """ProPublica COMPAS dataset."""
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    dest = os.path.join(DATA_DIR, "compas.csv")
    download_file(url, dest)
    df = pd.read_csv(dest)
    return df


def load_ieee_fraud():
    """IEEE Fraud Detection (requires Kaggle API or manual download)."""
    dest = os.path.join(DATA_DIR, "ieee fraud.csv")
    if os.path.exists(dest):
        return pd.read_csv(dest)
    print("IEEE Fraud dataset requires manual download from Kaggle.")
    print("https://www.kaggle.com/c/ieee-fraud-detection/data")
    print("Place 'train_transaction.csv' as 'ieee fraud.csv' in data/raw/")
    return None


def load_mimic():
    """MIMIC-III (requires PhysioNet credentialed access)."""
    dest = os.path.join(DATA_DIR, "mimic.csv")
    if os.path.exists(dest):
        return pd.read_csv(dest)
    print("MIMIC-III requires PhysioNet credentialed access.")
    print("https://physionet.org/content/mimiciii/1.4/")
    print("Place processed data as 'mimic.csv' in data/raw/")
    return None


if __name__ == "__main__":
    print("Downloading datasets...")
    load_adult()
    load_german()
    load_compas()
    load_ieee_fraud()
    load_mimic()
    print("Done.")
