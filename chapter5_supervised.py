#!/usr/bin/env python3
"""
chapter5_supervised.py

Supervised learning chapter script:
- Classification: predict Rating class (Low/Medium/High)
- Regression: predict NumOwned (log1p target) and GameWeight

Outputs:
  outputs_supervised/
    classification/
      classification_summary.csv
      confusion_matrix.png
      model_comparison.csv
      feature_importance_permutation.csv
    regression/
      regression_summary.csv
      regression_model_comparison.csv
      pred_vs_true_numowned.png
      pred_vs_true_gameweight.png
"""

import os
import re
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    """Compute RMSE without using mean_squared_error(..., squared=False)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names: keep original but also allow robust matching
    df.columns = [c.strip() for c in df.columns]
    return df


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first existing column name from candidates (case-insensitive match)."""
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def is_binary_series(s: pd.Series) -> bool:
    vals = pd.Series(s.dropna().unique())
    if len(vals) == 0:
        return False
    # allow bool, 0/1, "0"/"1"
    allowed = set([0, 1, True, False, "0", "1"])
    return set(vals.tolist()).issubset(allowed)


def coerce_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    # strings "0"/"1"
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).clip(0, 1)


# ----------------------------
# Dataset prep (consistent with your Chapter 3 logic)
# ----------------------------

@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_cols: List[str]
    numeric_cols: List[str]
    binary_cols: List[str]
    target_rating_col: str
    target_numowned_col: Optional[str]
    target_gameweight_col: Optional[str]


def clean_and_prepare(df_raw: pd.DataFrame) -> PreparedData:
    df = df_raw.copy()

    # Common columns (robustly locate)
    col_id = find_column(df, ["BGGId", "bggid"])
    col_name = find_column(df, ["Name"])
    col_desc = find_column(df, ["Description"])
    col_img = find_column(df, ["ImagePath", "imagepath"])
    col_goodplayers = find_column(df, ["GoodPlayers", "goodplayers"])
    col_family = find_column(df, ["Family"])

    # Targets
    rating_col = find_column(df, ["Rating", "RatingClass", "rating_class", "rating"])
    if rating_col is None:
        raise ValueError(
            "Could not find Rating target column. Expected a column like: Rating / RatingClass."
        )

    numowned_col = find_column(df, ["NumOwned", "numowned"])
    gameweight_col = find_column(df, ["GameWeight", "gameweight"])

    # Drop obvious non-feature columns
    drop_cols = [c for c in [col_id, col_name, col_desc, col_img, col_goodplayers, col_family] if c is not None]

    # Also drop rank columns by default to avoid leakage / placeholder issues
    rank_cols = [c for c in df.columns if c.lower().startswith("rank:") or c.lower().startswith("rank_")]
    drop_cols += rank_cols

    # Keep targets aside; do not drop them yet
    # We will build X by excluding targets and drop_cols
    # Basic sentinel handling (only if columns exist)
    # MaxPlayers: cap 999 -> 20 (as you described)
    maxp = find_column(df, ["MaxPlayers", "maxplayers"])
    if maxp is not None:
        df[maxp] = pd.to_numeric(df[maxp], errors="coerce")
        df.loc[df[maxp] == 999, maxp] = 20
        # 0 as unknown -> NaN to impute later
        df.loc[df[maxp] == 0, maxp] = np.nan

    minp = find_column(df, ["MinPlayers", "minplayers"])
    if minp is not None:
        df[minp] = pd.to_numeric(df[minp], errors="coerce")
        df.loc[df[minp] == 0, minp] = np.nan

    # GameWeight: 0 indicates "not rated" in your description -> treat as NaN
    if gameweight_col is not None:
        df[gameweight_col] = pd.to_numeric(df[gameweight_col], errors="coerce")
        df.loc[df[gameweight_col] == 0, gameweight_col] = np.nan

    # Manufacturer playtime cap
    mfg_play = find_column(df, ["MfgPlaytime", "mfgplaytime"])
    if mfg_play is not None:
        df[mfg_play] = pd.to_numeric(df[mfg_play], errors="coerce")
        df.loc[df[mfg_play] > 1440, mfg_play] = 1440
        df.loc[df[mfg_play] == 0, mfg_play] = np.nan

    # ComAgeRec: keep; will be imputed
    com_age = find_column(df, ["ComAgeRec", "comagerec"])
    if com_age is not None:
        df[com_age] = pd.to_numeric(df[com_age], errors="coerce")

    # Ensure NumOwned numeric
    if numowned_col is not None:
        df[numowned_col] = pd.to_numeric(df[numowned_col], errors="coerce")

    # Rating cleanup: accept Low/Medium/High in many formats
    df[rating_col] = df[rating_col].astype(str).str.strip()
    df[rating_col] = df[rating_col].str.lower()

    def normalize_rating(x: str) -> str:
        x = x.strip().lower()
        if x in ["low", "l"]:
            return "low"
        if x in ["medium", "med", "m", "mid"]:
            return "medium"
        if x in ["high", "h"]:
            return "high"
        # If it contains low/medium/high inside a longer string
        if "low" in x:
            return "low"
        if "medium" in x:
            return "medium"
        if "high" in x:
            return "high"
        return x

    df[rating_col] = df[rating_col].apply(normalize_rating)

    # Remove rows with unknown/unexpected rating labels
    valid = set(["low", "medium", "high"])
    before = len(df)
    df = df[df[rating_col].isin(valid)].copy()
    after = len(df)
    if after < before:
        print(f"[INFO] Dropped {before-after} rows with invalid Rating labels.")

    # Build initial candidate feature set
    exclude = set(drop_cols + [rating_col])
    # For regression tasks, targets must also be excluded from X (when training those targets)
    # We'll keep numowned/gameweight in df but exclude from X below if present.
    if numowned_col is not None:
        exclude.add(numowned_col)
    if gameweight_col is not None:
        exclude.add(gameweight_col)

    # Category flags: Cat:* (keep)
    # Kickstarted / IsReimplementation (keep)
    # All numeric gameplay, complexity, engagement variables (keep; transform later)
    feature_candidates = [c for c in df.columns if c not in exclude]

    # Identify binary vs numeric columns
    binary_cols = []
    numeric_cols = []
    for c in feature_candidates:
        if df[c].dtype == object:
            # Try interpret Cat:* or binary-like strings as binary
            if c.lower().startswith("cat:") or c.lower().startswith("cat_"):
                binary_cols.append(c)
            else:
                # If object but only 0/1 values -> treat as binary
                if is_binary_series(df[c]):
                    binary_cols.append(c)
                else:
                    # Non-numeric text-like -> drop from features (safety)
                    pass
        else:
            # numeric dtype
            if is_binary_series(df[c]):
                binary_cols.append(c)
            else:
                numeric_cols.append(c)

    # Ensure binary are 0/1 ints
    for c in binary_cols:
        df[c] = coerce_binary(df[c])

    # Force numeric
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Apply log1p to heavy-tailed count-like variables (if present)
    # This helps both supervised and clustering stability.
    heavy_tail_patterns = [
        r"^NumOwned$",
        r"^NumWish$",
        r"^NumWant$",
        r"^NumUserRatings$",
        r"^NumComments$",
        r"^NumAlternates$",
        r"^NumExpansions$",
        r"^NumImplementations$",
        r"^NumWeightVotes$",
    ]
    heavy_cols = []
    for c in numeric_cols:
        for pat in heavy_tail_patterns:
            if re.match(pat, c, flags=re.IGNORECASE):
                heavy_cols.append(c)
                break

    # Only transform if column is non-negative
    for c in heavy_cols:
        series = df[c]
        if series.dropna().min() >= 0:
            df[c] = np.log1p(series)

    # Final feature columns
    feature_cols = sorted(list(set(numeric_cols + binary_cols)))

    return PreparedData(
        df=df,
        feature_cols=feature_cols,
        numeric_cols=sorted(list(set(numeric_cols))),
        binary_cols=sorted(list(set(binary_cols))),
        target_rating_col=rating_col,
        target_numowned_col=numowned_col,
        target_gameweight_col=gameweight_col,
    )


def build_preprocessor(numeric_cols: List[str], binary_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # no scaling for binary
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("bin", binary_pipe, binary_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre


# ----------------------------
# Classification
# ----------------------------

def classification_section(prep: PreparedData, out_dir: str) -> None:
    ensure_dir(out_dir)

    df = prep.df.copy()
    X = df[prep.feature_cols].copy()
    y = df[prep.target_rating_col].copy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pre = build_preprocessor(prep.numeric_cols, prep.binary_cols)

    models = {
        "logreg": LogisticRegression(max_iter=2000, multi_class="auto"),
        "rf": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(random_state=42),
    }

    param_grids = {
        "logreg": {
            "clf__C": [0.3, 1.0, 3.0],
            "clf__class_weight": [None, "balanced"],
        },
        "rf": {
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_leaf": [1, 3, 5],
        },
        "hgb": {
            "clf__max_depth": [None, 6, 10],
            "clf__learning_rate": [0.05, 0.1],
        },
    }

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    rows = []
    best_models: Dict[str, Pipeline] = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        grid = GridSearchCV(
            pipe,
            param_grid=param_grids.get(name, {}),
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        best_models[name] = best

        y_pred = best.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        f1w = f1_score(y_test, y_pred, average="weighted")

        rows.append({
            "model": name,
            "best_params": str(grid.best_params_),
            "test_accuracy": acc,
            "test_f1_macro": f1m,
            "test_f1_weighted": f1w,
        })

    model_comparison = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
    save_csv(model_comparison, os.path.join(out_dir, "model_comparison.csv"))

    # Select best model (macro F1)
    best_name = model_comparison.iloc[0]["model"]
    best_model = best_models[best_name]
    y_pred = best_model.predict(X_test)

    # Detailed summary
    report_txt = classification_report(y_test, y_pred, digits=3)
    summary_df = pd.DataFrame([{
        "best_model": best_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "best_params": model_comparison.iloc[0]["best_params"],
    }])
    save_csv(summary_df, os.path.join(out_dir, "classification_summary.csv"))

    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    # Confusion matrix plot
    labels = ["low", "medium", "high"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix ({best_name})")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # Permutation importance (on test set, model-agnostic)
    # Note: can be slow; still manageable for ~20k rows.
    try:
        X_test_pre = X_test.copy()
        result = permutation_importance(
            best_model,
            X_test_pre,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="f1_macro",
            n_jobs=-1,
        )
        importances = pd.DataFrame({
            "feature": prep.feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)
        save_csv(importances, os.path.join(out_dir, "feature_importance_permutation.csv"))
    except Exception as e:
        with open(os.path.join(out_dir, "feature_importance_error.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))


# ----------------------------
# Regression
# ----------------------------

def regression_one_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    numeric_cols: List[str],
    binary_cols: List[str],
    target_col: str,
    out_dir: str,
    target_name: str,
    log_target: bool = False,
) -> pd.DataFrame:
    """Train and evaluate multiple regression models for one target."""
    ensure_dir(out_dir)

    data = df.copy()
    y = pd.to_numeric(data[target_col], errors="coerce")

    # Drop rows where target missing
    mask = y.notna()
    data = data.loc[mask].copy()
    y = y.loc[mask].copy()

    if log_target:
        # Only valid for non-negative targets
        y = np.log1p(y.clip(lower=0))

    X = data[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pre = build_preprocessor(numeric_cols, binary_cols)

    models = {
        "ridge": Ridge(random_state=42),
        "rf": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "hgb": HistGradientBoostingRegressor(random_state=42),
    }

    grids = {
        "ridge": {"reg__alpha": [0.3, 1.0, 3.0, 10.0]},
        "rf": {"reg__max_depth": [None, 10, 20], "reg__min_samples_leaf": [1, 3, 5]},
        "hgb": {"reg__learning_rate": [0.05, 0.1], "reg__max_depth": [None, 6, 10]},
    }

    rows = []
    best_models: Dict[str, Pipeline] = {}

    # Use simple CV (regression)
    for name, reg in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])
        grid = GridSearchCV(
            pipe,
            param_grid=grids.get(name, {}),
            scoring="neg_root_mean_squared_error" if hasattr(__import__("sklearn.metrics"), "mean_squared_error") else "neg_mean_squared_error",
            cv=4,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        best_models[name] = best

        y_pred = best.predict(X_test)

        rows.append({
            "target": target_name,
            "model": name,
            "best_params": str(grid.best_params_),
            "rmse": rmse(y_test, y_pred),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        })

    comp = pd.DataFrame(rows).sort_values("rmse", ascending=True)
    save_csv(comp, os.path.join(out_dir, f"{target_name}_model_comparison.csv"))

    # Best model by RMSE
    best_name = comp.iloc[0]["model"]
    best_model = best_models[best_name]
    y_pred = best_model.predict(X_test)

    # Save scatter plot
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, s=10)
    plt.title(f"Predicted vs True ({target_name}) | best={best_name}")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pred_vs_true_{target_name}.png"), dpi=200)
    plt.close(fig)

    # Return summary row
    summary = pd.DataFrame([{
        "target": target_name,
        "best_model": best_name,
        "best_params": comp.iloc[0]["best_params"],
        "rmse": rmse(y_test, y_pred),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "log_target": bool(log_target),
        "n_rows_used": int(len(X_train) + len(X_test)),
    }])
    save_csv(summary, os.path.join(out_dir, f"{target_name}_summary.csv"))

    return summary


def regression_section(prep: PreparedData, out_dir: str) -> None:
    ensure_dir(out_dir)

    df = prep.df.copy()

    # For regression, we must include targets in df (they exist), but remove them from feature_cols already done.
    summaries = []

    if prep.target_numowned_col is not None and prep.target_numowned_col in df.columns:
        summaries.append(
            regression_one_target(
                df=df,
                feature_cols=prep.feature_cols,
                numeric_cols=prep.numeric_cols,
                binary_cols=prep.binary_cols,
                target_col=prep.target_numowned_col,
                out_dir=out_dir,
                target_name="numowned",
                log_target=True,  # recommended (heavy-tailed)
            )
        )

    if prep.target_gameweight_col is not None and prep.target_gameweight_col in df.columns:
        summaries.append(
            regression_one_target(
                df=df,
                feature_cols=prep.feature_cols,
                numeric_cols=prep.numeric_cols,
                binary_cols=prep.binary_cols,
                target_col=prep.target_gameweight_col,
                out_dir=out_dir,
                target_name="gameweight",
                log_target=False,
            )
        )

    if len(summaries) == 0:
        raise ValueError("No regression targets found (expected NumOwned and/or GameWeight columns).")

    all_summary = pd.concat(summaries, ignore_index=True)
    save_csv(all_summary, os.path.join(out_dir, "regression_summary.csv"))


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV (e.g., DM1.csv)")
    parser.add_argument("--out", required=True, help="Output folder (e.g., outputs_supervised)")
    args = parser.parse_args()

    out_root = args.out
    ensure_dir(out_root)

    # Reduce noisy warnings (but keep critical errors)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df_raw = safe_read_csv(args.input)
    prep = clean_and_prepare(df_raw)

    # Save quick info
    with open(os.path.join(out_root, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Input file: {args.input}\n")
        f.write(f"Rows used after Rating cleanup: {len(prep.df)}\n")
        f.write(f"Features: {len(prep.feature_cols)}\n")
        f.write(f"Numeric features: {len(prep.numeric_cols)}\n")
        f.write(f"Binary features: {len(prep.binary_cols)}\n")
        f.write(f"Rating target: {prep.target_rating_col}\n")
        f.write(f"Regression targets: NumOwned={prep.target_numowned_col}, GameWeight={prep.target_gameweight_col}\n")

    classification_section(prep, os.path.join(out_root, "classification"))
    regression_section(prep, os.path.join(out_root, "regression"))

    print(f"DONE. All supervised outputs saved to: {out_root}")


if __name__ == "__main__":
    main()
