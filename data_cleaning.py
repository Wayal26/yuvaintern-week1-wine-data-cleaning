"""
YuvaIntern Week 1 - Data Acquisition, Cleaning and Preprocessing
Dataset: UCI Wine Recognition dataset (via scikit-learn's packaged copy)

Run:
    pip install pandas numpy matplotlib scikit-learn
    python src/data_cleaning.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"

# Data acquisition
wine = load_wine(as_frame=True)
df = wine.frame.rename(columns={"target": "wine_class"})
df.to_csv(DATA_DIR / "wine_raw.csv", index=False)

print("Raw shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

feature_cols = [c for c in df.columns if c != "wine_class"]

# Domain validation
invalid_rows = set()
for col in feature_cols:
    invalid_rows.update(df.index[df[col] <= 0].tolist())

print("Invalid non-positive feature rows:", len(invalid_rows))

# IQR-based outlier detection and capping
clean = df.copy()
for col in feature_cols:
    q1 = clean[col].quantile(0.25)
    q3 = clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((clean[col] < lower) | (clean[col] > upper)).sum()
    print(f"{col}: {outliers} IQR outliers; bounds=({lower:.3f}, {upper:.3f})")

    # Cap extreme values while retaining observations.
    clean[col] = clean[col].clip(lower=lower, upper=upper)

clean.to_csv(DATA_DIR / "wine_cleaned.csv", index=False)

print("\nCleaned shape:", clean.shape)
print("Cleaned dataset saved to:", DATA_DIR / "wine_cleaned.csv")
