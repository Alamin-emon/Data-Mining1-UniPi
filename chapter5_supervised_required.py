#!/usr/bin/env python3
"""
chapter5_supervised_required.py

DM1 — Chapter 5 (Classification + Regression) aligned with course requirements.

Classification (mandatory): Rating (Low/Medium/High) using:
  - Decision Tree
  - KNN
  - Naive Bayes

Regression (assignment-style):
  - Single regression: choose ONE target and ONE independent variable (auto-chosen by abs(correlation) on train split)
  - Multiple regression: same target with 2+ predictors using Linear + at least 2 non-linear approaches

Outputs:
  <out>/
    run_info.txt
    classification/
      model_comparison.csv
      classification_summary.csv
      classification_report.txt
      confusion_matrix.png               (best model)
      roc_curve.png                     (best model, multiclass OvR)
      confusion_matrix_<model>.png       (all models)
      roc_curve_<model>.png              (all models)
      feature_importance_permutation.csv (best model; optional)
    regression/
      regression_summary.csv
      single_linear_summary.csv
      single_linear_pred_vs_true.png
      multiple_model_comparison.csv
      pred_vs_true_<best_model>.png

Run:
  python3 chapter5_supervised_required.py --input DM1_game_dataset.csv --out outputs_supervised
  python3 chapter5_supervised_required.py --input DM1_game_dataset.csv --out outputs_supervised --reg_target NumOwned
"""

import os
import re
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
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
    allowed = {0, 1, True, False, "0", "1"}
    return set(vals.tolist()).issubset(allowed)


def coerce_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).clip(0, 1)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ----------------------------
# Dataset prep (lightweight, consistent with your Chapter 3 logic)
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

    # Non-features
    drop_text_like = [
        find_column(df, ["BGGId", "bggid"]),
        find_column(df, ["Name"]),
        find_column(df, ["Description"]),
        find_column(df, ["ImagePath", "imagepath"]),
        find_column(df, ["GoodPlayers", "goodplayers"]),
        find_column(df, ["Family"]),
    ]
    drop_cols = [c for c in drop_text_like if c is not None]

    # Drop rank columns (leakage / post-hoc)
    rank_cols = [c for c in df.columns if c.lower().startswith("rank:") or c.lower().startswith("rank_")]
    drop_cols += rank_cols

    # Targets
    rating_col = find_column(df, ["Rating", "RatingClass", "rating_class", "rating"])
    if rating_col is None:
        raise ValueError("Could not find Rating target column (expected Rating / RatingClass).")

    numowned_col = find_column(df, ["NumOwned", "numowned"])
    gameweight_col = find_column(df, ["GameWeight", "gameweight"])

    # Sentinel handling (if present)
    maxp = find_column(df, ["MaxPlayers", "maxplayers"])
    if maxp is not None:
        df[maxp] = pd.to_numeric(df[maxp], errors="coerce")
        df.loc[df[maxp] == 999, maxp] = 20
        df.loc[df[maxp] == 0, maxp] = np.nan

    minp = find_column(df, ["MinPlayers", "minplayers"])
    if minp is not None:
        df[minp] = pd.to_numeric(df[minp], errors="coerce")
        df.loc[df[minp] == 0, minp] = np.nan

    if gameweight_col is not None:
        df[gameweight_col] = pd.to_numeric(df[gameweight_col], errors="coerce")
        df.loc[df[gameweight_col] == 0, gameweight_col] = np.nan

    mfg_play = find_column(df, ["MfgPlaytime", "mfgplaytime"])
    if mfg_play is not None:
        df[mfg_play] = pd.to_numeric(df[mfg_play], errors="coerce")
        df.loc[df[mfg_play] > 1440, mfg_play] = 1440
        df.loc[df[mfg_play] == 0, mfg_play] = np.nan

    # Ensure numeric targets
    if numowned_col is not None:
        df[numowned_col] = pd.to_numeric(df[numowned_col], errors="coerce")

    # Rating cleanup -> low/medium/high
    df[rating_col] = df[rating_col].astype(str).str.strip().str.lower()

    def normalize_rating(x: str) -> str:
        x = x.strip().lower()
        if x in ["low", "l"] or "low" in x:
            return "low"
        if x in ["medium", "med", "m", "mid"] or "medium" in x:
            return "medium"
        if x in ["high", "h"] or "high" in x:
            return "high"
        return x

    df[rating_col] = df[rating_col].apply(normalize_rating)

    valid = {"low", "medium", "high"}
    before = len(df)
    df = df[df[rating_col].isin(valid)].copy()
    if len(df) < before:
        print(f"[INFO] Dropped {before - len(df)} rows with invalid Rating labels.")

    # Build feature candidates (exclude non-features + targets)
    exclude = set(drop_cols + [rating_col])
    if numowned_col is not None:
        exclude.add(numowned_col)
    if gameweight_col is not None:
        exclude.add(gameweight_col)

    feature_candidates = [c for c in df.columns if c not in exclude]

    # Split into numeric/binary
    binary_cols: List[str] = []
    numeric_cols: List[str] = []

    for c in feature_candidates:
        if df[c].dtype == object:
            # Keep category flags
            if c.lower().startswith("cat:") or c.lower().startswith("cat_"):
                binary_cols.append(c)
            elif is_binary_series(df[c]):
                binary_cols.append(c)
            else:
                # Drop other text-like fields (safety)
                continue
        else:
            if is_binary_series(df[c]):
                binary_cols.append(c)
            else:
                numeric_cols.append(c)

    # Force types
    for c in binary_cols:
        df[c] = coerce_binary(df[c])

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Log1p heavy-tailed count-like features (features only)
    heavy_tail_patterns = [
        r"^NumOwned$", r"^NumWish$", r"^NumWant$", r"^NumUserRatings$",
        r"^NumComments$", r"^NumAlternates$", r"^NumExpansions$",
        r"^NumImplementations$", r"^NumWeightVotes$",
    ]
    heavy_cols = []
    for c in numeric_cols:
        for pat in heavy_tail_patterns:
            if re.match(pat, c, flags=re.IGNORECASE):
                heavy_cols.append(c)
                break

    for c in heavy_cols:
        s = df[c]
        if s.dropna().empty:
            continue
        if s.dropna().min() >= 0:
            df[c] = np.log1p(s)

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
    # For KNN, binary scaling is often useful too. Scaling 0/1 does not hurt.
    binary_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("bin", binary_pipe, binary_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


# ----------------------------
# Classification (DT, KNN, NB) + Confusion Matrix + ROC
# ----------------------------

def plot_confusion(cm: np.ndarray, labels: List[str], title: str, path: str) -> None:
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def plot_multiclass_roc(y_true: pd.Series, y_proba: np.ndarray, classes: List[str], title: str, path: str) -> None:
    """
    Multiclass ROC using One-vs-Rest.
    Produces per-class curves + micro-average.
    """
    y_bin = label_binarize(y_true, classes=classes)
    fig = plt.figure(figsize=(7, 6))

    # Per-class ROC
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro (AUC={auc_micro:.3f})")

    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def classification_section(prep: PreparedData, out_dir: str) -> None:
    ensure_dir(out_dir)

    df = prep.df.copy()
    X = df[prep.feature_cols].copy()
    y = df[prep.target_rating_col].copy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = build_preprocessor(prep.numeric_cols, prep.binary_cols)

    models = {
        "dtree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(),
        "nb": GaussianNB(),
    }

    param_grids = {
        "dtree": {
            "clf__criterion": ["gini", "entropy", "log_loss"],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__class_weight": [None, "balanced"],
            "clf__ccp_alpha": [0.0, 0.001, 0.01],
        },
        "knn": {
            "clf__n_neighbors": [3, 5, 7, 11, 15, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["minkowski"],
            "clf__p": [1, 2],  # Manhattan / Euclidean
        },
        "nb": {
            "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        },
    }

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    classes = ["low", "medium", "high"]

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
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

        rows.append({
            "model": name,
            "best_params": str(grid.best_params_),
            "test_accuracy": acc,
            "test_precision_macro": float(p),
            "test_recall_macro": float(r),
            "test_f1_macro": f1m,
            "test_f1_weighted": f1w,
        })

        # Save per-model plots
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        plot_confusion(
            cm, classes,
            title=f"Confusion Matrix ({name})",
            path=os.path.join(out_dir, f"confusion_matrix_{name}.png"),
        )

        # ROC curve (needs predict_proba)
        try:
            y_proba = best.predict_proba(X_test)
            plot_multiclass_roc(
                y_true=y_test,
                y_proba=y_proba,
                classes=classes,
                title=f"ROC Curve OvR ({name})",
                path=os.path.join(out_dir, f"roc_curve_{name}.png"),
            )
        except Exception as e:
            with open(os.path.join(out_dir, f"roc_curve_{name}_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))

    model_comparison = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
    save_csv(model_comparison, os.path.join(out_dir, "model_comparison.csv"))

    # Best model by macro F1
    best_name = model_comparison.iloc[0]["model"]
    best_model = best_models[best_name]
    y_pred = best_model.predict(X_test)

    # Summary + report
    summary_df = pd.DataFrame([{
        "best_model": best_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": float(precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)[0]),
        "recall_macro": float(precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)[1]),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "best_params": model_comparison.iloc[0]["best_params"],
    }])
    save_csv(summary_df, os.path.join(out_dir, "classification_summary.csv"))

    report_txt = classification_report(y_test, y_pred, digits=3, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plot_confusion(
        cm, classes,
        title=f"Confusion Matrix (best={best_name})",
        path=os.path.join(out_dir, "confusion_matrix.png"),
    )

    try:
        y_proba = best_model.predict_proba(X_test)
        plot_multiclass_roc(
            y_true=y_test,
            y_proba=y_proba,
            classes=classes,
            title=f"ROC Curve OvR (best={best_name})",
            path=os.path.join(out_dir, "roc_curve.png"),
        )
    except Exception as e:
        with open(os.path.join(out_dir, "roc_curve_error.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))

    # Optional: permutation importance for best model (model-agnostic)
    try:
        result = permutation_importance(
            best_model,
            X_test,
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
# Regression (single + multiple, one target)
# ----------------------------

def choose_regression_target(prep: PreparedData, requested: Optional[str]) -> str:
    if requested:
        if requested not in prep.df.columns:
            raise ValueError(f"--reg_target='{requested}' not found in dataset columns.")
        return requested

    # Default preference: NumOwned if present, else GameWeight
    if prep.target_numowned_col is not None and prep.target_numowned_col in prep.df.columns:
        return prep.target_numowned_col
    if prep.target_gameweight_col is not None and prep.target_gameweight_col in prep.df.columns:
        return prep.target_gameweight_col

    raise ValueError("No regression target found (expected NumOwned and/or GameWeight).")


def single_linear_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    out_dir: str,
    log_target: bool,
) -> pd.DataFrame:
    """
    Single regression: ONE X column + linear regression.
    Automatically selects the single numeric predictor with highest |correlation| on train split.
    """
    ensure_dir(out_dir)

    data = df.copy()
    y = pd.to_numeric(data[target_col], errors="coerce")

    # Drop missing target
    mask = y.notna()
    data = data.loc[mask].copy()
    y = y.loc[mask].copy()

    if log_target:
        y = np.log1p(y.clip(lower=0))

    # Only numeric candidates for single-variable linear regression
    numeric_candidates = []
    for c in feature_cols:
        s = pd.to_numeric(data[c], errors="coerce")
        if s.notna().sum() > 0:
            numeric_candidates.append(c)

    if len(numeric_candidates) == 0:
        raise ValueError("No usable numeric predictors found for single-variable regression.")

    # Split
    X_all = data[numeric_candidates].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42
    )

    # Choose best single predictor by absolute correlation on train (after median impute)
    corrs = []
    for c in numeric_candidates:
        xt = pd.to_numeric(X_train[c], errors="coerce")
        xt = xt.fillna(xt.median())
        if xt.nunique() <= 1:
            continue
        corr = np.corrcoef(xt.values, y_train.values)[0, 1]
        if np.isfinite(corr):
            corrs.append((c, abs(float(corr)), float(corr)))

    if len(corrs) == 0:
        # fallback: first numeric
        best_feat = numeric_candidates[0]
        best_abs, best_signed = np.nan, np.nan
    else:
        corrs.sort(key=lambda t: t[1], reverse=True)
        best_feat, best_abs, best_signed = corrs[0]

    # Build pipeline: impute + scale + linear regression
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), [best_feat]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    model = Pipeline(steps=[("pre", pre), ("reg", LinearRegression())])

    model.fit(X_train[[best_feat]], y_train)
    y_pred = model.predict(X_test[[best_feat]])

    # Plot
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, s=10)
    plt.title(f"Single Linear Regression | y={target_col} | x={best_feat}")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "single_linear_pred_vs_true.png"), dpi=200)
    plt.close(fig)

    summary = pd.DataFrame([{
        "target": target_col,
        "single_feature": best_feat,
        "abs_corr_on_train": best_abs,
        "signed_corr_on_train": best_signed,
        "rmse": rmse(y_test, y_pred),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "log_target": bool(log_target),
        "n_rows_used": int(len(X_train) + len(X_test)),
    }])
    save_csv(summary, os.path.join(out_dir, "single_linear_summary.csv"))
    return summary


def multiple_regression_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    numeric_cols: List[str],
    binary_cols: List[str],
    target_col: str,
    out_dir: str,
    log_target: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multiple regression: 2+ predictors with linear + >=2 non-linear.
    """
    ensure_dir(out_dir)

    data = df.copy()
    y = pd.to_numeric(data[target_col], errors="coerce")

    mask = y.notna()
    data = data.loc[mask].copy()
    y = y.loc[mask].copy()

    if log_target:
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

    for name, reg in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])
        grid = GridSearchCV(
            pipe,
            param_grid=grids.get(name, {}),
            scoring="neg_root_mean_squared_error",
            cv=4,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        best_models[name] = best

        y_pred = best.predict(X_test)

        rows.append({
            "target": target_col,
            "model": name,
            "best_params": str(grid.best_params_),
            "rmse": rmse(y_test, y_pred),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        })

    comp = pd.DataFrame(rows).sort_values("rmse", ascending=True)
    save_csv(comp, os.path.join(out_dir, "multiple_model_comparison.csv"))

    # Best model scatter plot
    best_name = comp.iloc[0]["model"]
    best_model = best_models[best_name]
    y_pred = best_model.predict(X_test)

    fig = plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, s=10)
    plt.title(f"Multiple Regression | y={target_col} | best={best_name}")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pred_vs_true_{best_name}.png"), dpi=200)
    plt.close(fig)

    summary = pd.DataFrame([{
        "target": target_col,
        "best_model": best_name,
        "best_params": comp.iloc[0]["best_params"],
        "rmse": rmse(y_test, y_pred),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "log_target": bool(log_target),
        "n_rows_used": int(len(X_train) + len(X_test)),
        "n_features_used": int(len(feature_cols)),
    }])
    return comp, summary


def regression_section(prep: PreparedData, out_dir: str, reg_target: Optional[str]) -> None:
    ensure_dir(out_dir)

    df = prep.df.copy()
    target_col = choose_regression_target(prep, reg_target)

    # Heuristic: log1p for NumOwned-like targets
    log_target = target_col.lower() in {"numowned", "numwish", "numwant", "numuserratings", "numcomments"}

    single_summary = single_linear_regression(
        df=df,
        feature_cols=prep.feature_cols,
        target_col=target_col,
        out_dir=out_dir,
        log_target=log_target,
    )

    comp, multi_summary = multiple_regression_models(
        df=df,
        feature_cols=prep.feature_cols,
        numeric_cols=prep.numeric_cols,
        binary_cols=prep.binary_cols,
        target_col=target_col,
        out_dir=out_dir,
        log_target=log_target,
    )

    all_summary = pd.concat([single_summary.assign(section="single_linear"),
                             multi_summary.assign(section="multiple_models")],
                            ignore_index=True)
    save_csv(all_summary, os.path.join(out_dir, "regression_summary.csv"))


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV (e.g., DM1_game_dataset.csv)")
    parser.add_argument("--out", required=True, help="Output folder (e.g., outputs_supervised)")
    parser.add_argument("--reg_target", default=None, help="Regression target column (default: NumOwned if present else GameWeight)")
    args = parser.parse_args()

    out_root = args.out
    ensure_dir(out_root)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df_raw = safe_read_csv(args.input)
    prep = clean_and_prepare(df_raw)

    with open(os.path.join(out_root, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Input file: {args.input}\n")
        f.write(f"Rows used after Rating cleanup: {len(prep.df)}\n")
        f.write(f"Features: {len(prep.feature_cols)}\n")
        f.write(f"Numeric features: {len(prep.numeric_cols)}\n")
        f.write(f"Binary features: {len(prep.binary_cols)}\n")
        f.write(f"Rating target: {prep.target_rating_col}\n")
        f.write(f"Regression targets present: NumOwned={prep.target_numowned_col}, GameWeight={prep.target_gameweight_col}\n")
        f.write(f"Requested regression target: {args.reg_target}\n")

    classification_section(prep, os.path.join(out_root, "classification"))
    regression_section(prep, os.path.join(out_root, "regression"), args.reg_target)

    print(f"DONE. All outputs saved to: {out_root}")


if __name__ == "__main__":
    main()
