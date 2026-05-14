"""
DM1 — Data Understanding & Preparation (BGG dataset)
Tailored to: DM1_game_dataset.csv (21925 rows, 46 cols)

Creates:
- Table 2.1: dataset glance
- Table 3.1: key-variable data dictionary (auto + notes placeholders)
- Table 3.2: descriptive stats for key numeric vars (count, miss%, mean/median/std/min/max/skew)
- Table 3.3: missingness + handling decisions (pre-filled for Family/LanguageEase/ComAgeRec/etc.)
- Table 3.4: high-correlation pairs (|r| > threshold) + drop-list per your policy
- Figures 3.1–3.6: univariate + pairwise + correlation heatmap
- Cleaned datasets:
    1) DM1_clean_before_drops.csv  (after sentinel fixes + imputations)
    2) DM1_prepared_after_drops.csv (after redundancy elimination)
    3) DM1_prepared_with_logs.csv  (adds log1p columns for heavy-tailed metrics)

Run:
  python dm1_ch3_prep.py --input /mnt/data/DM1_game_dataset.csv --outdir outputs --corr-threshold 0.85
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional skewness improvement if SciPy is installed
try:
    from scipy.stats import skew as scipy_skew
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -------------------------
# Helpers
# -------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def skewness(x: pd.Series) -> float:
    v = x.dropna().values
    if len(v) < 3:
        return np.nan
    if HAS_SCIPY:
        return float(scipy_skew(v, bias=False))
    # fallback (not perfectly unbiased)
    m = np.mean(v)
    sd = np.std(v, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(((v - m) / sd) ** 3))

def detect_prefix_cols(cols, prefix: str):
    return [c for c in cols if c.startswith(prefix)]

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -------------------------
# Tables
# -------------------------
def table_dataset_glance(df: pd.DataFrame, input_file: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"Property": "File", "Value": os.path.basename(input_file)},
        {"Property": "Rows", "Value": df.shape[0]},
        {"Property": "Columns", "Value": df.shape[1]},
        {"Property": "Unit of observation", "Value": "One board game"},
        {"Property": "Unique identifier", "Value": "BGGId"},
        {"Property": "Target variable", "Value": "Rating (Low/Medium/High)"},
    ])

def table_data_dictionary(df: pd.DataFrame, variables) -> pd.DataFrame:
    out = []
    n = len(df)
    for v in variables:
        if v not in df.columns:
            continue
        s = df[v]
        miss_pct = (s.isna().sum() / n) * 100

        if pd.api.types.is_numeric_dtype(s):
            sn = safe_numeric(s)
            rng = f"{np.nanmin(sn.values):g} to {np.nanmax(sn.values):g}"
            vtype = str(sn.dtype)
        else:
            rng = "Categorical/Text"
            vtype = str(s.dtype)

        out.append({
            "Variable": v,
            "Type": vtype,
            "Meaning": "",   # keep empty; you fill in report text
            "Range": rng,
            "Missing%": round(miss_pct, 2),
            "Notes": ""      # e.g. "0 used as unknown", "999=unlimited", etc.
        })
    return pd.DataFrame(out)

def table_descriptive_stats(df: pd.DataFrame, variables) -> pd.DataFrame:
    n = len(df)
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        x = safe_numeric(df[v])
        miss = x.isna().sum()
        rows.append({
            "Variable": v,
            "Count": int(x.notna().sum()),
            "Miss%": round((miss / n) * 100, 2),
            "Mean": round(float(x.mean(skipna=True)), 4) if x.notna().any() else np.nan,
            "Median": round(float(x.median(skipna=True)), 4) if x.notna().any() else np.nan,
            "Std": round(float(x.std(skipna=True, ddof=1)), 4) if x.notna().any() else np.nan,
            "Min": float(x.min(skipna=True)) if x.notna().any() else np.nan,
            "Max": float(x.max(skipna=True)) if x.notna().any() else np.nan,
            "Skew": round(skewness(x), 4),
        })
    return pd.DataFrame(rows)

def table_missingness_with_decisions(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    miss_pct = (df.isna().sum() / n) * 100
    t = (
        pd.DataFrame({"Variable": miss_pct.index, "Missing%": miss_pct.values})
        .sort_values("Missing%", ascending=False)
        .reset_index(drop=True)
    )

    # Pre-fill decisions exactly aligned to your write-up
    decision_map = {
        "Family": ("Exclude from modeling", "69% missing; optional metadata; too sparse."),
        "LanguageEase": ("Exclude (default) or median-impute if needed", "26.9% missing; exclude unless required."),
        "ComAgeRec": ("Median imputation + missing indicator", "25.2% missing; symmetric enough for median."),
        "ImagePath": ("Keep (but excluded from numeric modeling)", "URL field; negligible missingness."),
        "Description": ("Exclude from numeric modeling", "Text field; could be used for NLP in future."),
        "Name": ("Exclude from numeric modeling", "Identifier text only."),
        "GoodPlayers": ("Exclude", "List-like strings; inconsistent formatting."),
    }

    t["Decision"] = ""
    t["Justification"] = ""
    for i, r in t.iterrows():
        var = r["Variable"]
        if var in decision_map:
            t.at[i, "Decision"] = decision_map[var][0]
            t.at[i, "Justification"] = decision_map[var][1]
    return t


# -------------------------
# Cleaning rules (as in your Chapter 3)
# -------------------------
def apply_sentinel_and_outlier_policy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements your stated policies:
    - MinPlayers == 0 -> unknown -> replace with median (non-zero)
    - MaxPlayers == 0 -> unknown -> replace with median (excluding 0 and 999)
    - MaxPlayers == 999 -> unlimited -> cap to 20
    - GameWeight == 0 -> unrated -> impute with median + indicator
    - MfgPlaytime == 0 -> missing -> impute with median + cap >1440 to 1440
    - ComAgeRec -> median impute + indicator (true NaNs)
    - Drop LanguageEase by default later (not here)
    """
    d = df.copy()

    # Ensure numeric types for relevant columns
    numeric_cols = [
        "YearPublished", "GameWeight", "ComWeight",
        "MinPlayers", "MaxPlayers", "ComAgeRec", "LanguageEase",
        "NumOwned", "NumWant", "NumWish", "NumUserRatings", "NumWeightVotes",
        "MfgPlaytime", "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec",
        "NumAlternates", "NumExpansions", "NumImplementations", "NumComments"
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = safe_numeric(d[c])

    # MinPlayers: 0 means unknown
    if "MinPlayers" in d.columns:
        mp = d["MinPlayers"].copy()
        mp[mp == 0] = np.nan
        mp_med = mp.median(skipna=True)
        d["MinPlayers"] = mp.fillna(mp_med)

    # MaxPlayers: 0 unknown; 999 unlimited
    if "MaxPlayers" in d.columns:
        mx = d["MaxPlayers"].copy()
        mx[mx == 0] = np.nan
        mx[mx == 999] = 20
        mx_med = mx.median(skipna=True)
        d["MaxPlayers"] = mx.fillna(mx_med)

    # GameWeight: 0 means unrated
    if "GameWeight" in d.columns:
        d["GameWeight_missing"] = (d["GameWeight"] == 0).astype(int)
        gw = d["GameWeight"].copy()
        gw[gw == 0] = np.nan
        gw_med = gw.median(skipna=True)
        d["GameWeight"] = gw.fillna(gw_med)

    # MfgPlaytime: 0 missing; cap > 1440
    if "MfgPlaytime" in d.columns:
        d["MfgPlaytime_missing"] = (d["MfgPlaytime"] == 0).astype(int)
        pt = d["MfgPlaytime"].copy()
        pt[pt == 0] = np.nan
        pt = pt.clip(upper=1440)
        pt_med = pt.median(skipna=True)
        d["MfgPlaytime"] = pt.fillna(pt_med)

    # ComAgeRec: median impute + indicator (NaNs are true missing)
    if "ComAgeRec" in d.columns:
        d["ComAgeRec_missing"] = d["ComAgeRec"].isna().astype(int)
        ca_med = d["ComAgeRec"].median(skipna=True)
        d["ComAgeRec"] = d["ComAgeRec"].fillna(ca_med)

    return d


def add_log1p_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds log1p transforms for heavy-tailed engagement counts.
    Keeps raw columns and adds *_log1p columns.
    """
    d = df.copy()
    heavy_tail = [
        "NumOwned", "NumWish", "NumWant", "NumUserRatings",
        "NumExpansions", "NumAlternates", "NumImplementations", "NumComments"
    ]
    for c in heavy_tail:
        if c in d.columns:
            d[f"{c}_log1p"] = np.log1p(d[c].clip(lower=0))
    return d


def drop_non_model_fields(df: pd.DataFrame, drop_languageease: bool = True) -> pd.DataFrame:
    """
    Drops fields you excluded from modeling:
    - Text/identifier: BGGId, Name, Description, ImagePath
    - Sparse or special: Family, GoodPlayers
    - LanguageEase dropped by default (per your policy)
    Keeps: Rating (target), Cat:* flags, Rank:* columns (unless later dropped).
    """
    d = df.copy()
    to_drop = ["BGGId", "Name", "Description", "ImagePath", "Family", "GoodPlayers"]
    if drop_languageease and "LanguageEase" in d.columns:
        to_drop.append("LanguageEase")
    d = d.drop(columns=[c for c in to_drop if c in d.columns], errors="ignore")
    return d


# -------------------------
# Correlation analysis + redundancy elimination
# -------------------------
def high_corr_pairs(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame(columns=["Variable A", "Variable B", "Correlation"])
    corr = num.corr(numeric_only=True)

    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > threshold:
                pairs.append({"Variable A": cols[i], "Variable B": cols[j], "Correlation": float(r)})

    out = pd.DataFrame(pairs)
    if not out.empty:
        out = out.sort_values("Correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return out


def apply_redundancy_drops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns explicitly stated in your Table 3.4 / text:
    - NumUserRatings (redundant with NumOwned)
    - NumWant (redundant with NumWish)
    - NumWeightVotes (redundant with engagement metrics)
    - ComMinPlaytime and ComMaxPlaytime (redundant with MfgPlaytime)
    - ComWeight (near-identical to GameWeight)
    """
    d = df.copy()
    drop_list = [
        "NumUserRatings", "NumWant", "NumWeightVotes",
        "ComMinPlaytime", "ComMaxPlaytime", "ComWeight"
    ]
    d = d.drop(columns=[c for c in drop_list if c in d.columns], errors="ignore")
    return d


# -------------------------
# Figures (3.1–3.6)
# -------------------------
def fig_year_distribution(df: pd.DataFrame, outdir: str):
    y = df.get("YearPublished")
    if y is None:
        return
    y = safe_numeric(y)
    yf = y[(y >= 1900) & (y <= 2021)]
    plt.figure(figsize=(8, 4))
    plt.hist(yf.dropna(), bins=40)
    plt.title("Figure 3.1 — YearPublished distribution (1900–2021)")
    plt.xlabel("YearPublished")
    plt.ylabel("Count")
    save_fig(os.path.join(outdir, "fig_3_1_yearpublished_1900_2021.png"))

def fig_gameweight_distribution(df: pd.DataFrame, outdir: str):
    gw = df.get("GameWeight")
    if gw is None:
        return
    gw = safe_numeric(gw)
    plt.figure(figsize=(8, 4))
    plt.hist(gw.dropna(), bins=30)
    plt.title("Figure 3.2 — GameWeight distribution (after handling unrated=0)")
    plt.xlabel("GameWeight")
    plt.ylabel("Count")
    save_fig(os.path.join(outdir, "fig_3_2_gameweight.png"))

def fig_numowned_log1p(df: pd.DataFrame, outdir: str):
    if "NumOwned_log1p" not in df.columns and "NumOwned" not in df.columns:
        return
    x = df["NumOwned_log1p"] if "NumOwned_log1p" in df.columns else np.log1p(safe_numeric(df["NumOwned"]).clip(lower=0))
    plt.figure(figsize=(8, 4))
    plt.hist(pd.Series(x).dropna(), bins=40)
    plt.title("Figure 3.3 — NumOwned after log1p transform")
    plt.xlabel("log1p(NumOwned)")
    plt.ylabel("Count")
    save_fig(os.path.join(outdir, "fig_3_3_numowned_log1p.png"))

def fig_gw_vs_nur_log1p(df: pd.DataFrame, outdir: str):
    # Your report used GameWeight vs log-transformed NumUserRatings
    if "GameWeight" not in df.columns or "NumUserRatings" not in df.columns:
        return
    tmp = pd.DataFrame({
        "GameWeight": safe_numeric(df["GameWeight"]),
        "NumUserRatings_log1p": np.log1p(safe_numeric(df["NumUserRatings"]).clip(lower=0))
    }).dropna()

    plt.figure(figsize=(6, 5))
    plt.scatter(tmp["GameWeight"], tmp["NumUserRatings_log1p"], s=8)
    plt.title("Figure 3.4 — GameWeight vs log1p(NumUserRatings)")
    plt.xlabel("GameWeight")
    plt.ylabel("log1p(NumUserRatings)")
    save_fig(os.path.join(outdir, "fig_3_4_gw_vs_numuserratings_log1p.png"))

def fig_log1p_effect(df_raw: pd.DataFrame, outdir: str):
    # Figure 3.5: show original vs log1p (two separate images to keep it simple)
    if "NumOwned" not in df_raw.columns:
        return
    x = safe_numeric(df_raw["NumOwned"]).clip(lower=0)

    plt.figure(figsize=(8, 4))
    plt.hist(x.dropna(), bins=60)
    plt.title("Figure 3.5a — NumOwned (original scale)")
    plt.xlabel("NumOwned")
    plt.ylabel("Count")
    save_fig(os.path.join(outdir, "fig_3_5a_numowned_original.png"))

    xl = np.log1p(x)
    plt.figure(figsize=(8, 4))
    plt.hist(xl.dropna(), bins=60)
    plt.title("Figure 3.5b — NumOwned (log1p scale)")
    plt.xlabel("log1p(NumOwned)")
    plt.ylabel("Count")
    save_fig(os.path.join(outdir, "fig_3_5b_numowned_log1p.png"))

def fig_corr_heatmap(df: pd.DataFrame, outdir: str):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return
    corr = num.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Figure 3.6 — Correlation heatmap (numeric features)")
    save_fig(os.path.join(outdir, "fig_3_6_corr_heatmap.png"))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to DM1_game_dataset.csv")
    ap.add_argument("--outdir", default="outputs", help="Directory for tables/figures/CSVs")
    ap.add_argument("--corr-threshold", type=float, default=0.85, help="|r| threshold for Table 3.4")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    # Load
    df = pd.read_csv(args.input, low_memory=False)

    # Identify groups
    cat_cols = detect_prefix_cols(df.columns.tolist(), "Cat:")
    rank_cols = detect_prefix_cols(df.columns.tolist(), "Rank:")

    # Table 2.1
    save_csv(table_dataset_glance(df, args.input), os.path.join(args.outdir, "table_2_1_dataset_glance.csv"))

    # Table 3.3 (missingness + decisions)
    save_csv(table_missingness_with_decisions(df), os.path.join(args.outdir, "table_3_3_missingness_decisions.csv"))

    # Apply cleaning + imputations per your policy
    df_clean = apply_sentinel_and_outlier_policy(df)

    # Save intermediate cleaned dataset (before drops)
    save_csv(df_clean, os.path.join(args.outdir, "DM1_clean_before_drops.csv"))

    # Table 3.2 on your key numeric set
    key_numeric = [
        "YearPublished", "GameWeight", "MinPlayers", "MaxPlayers", "ComAgeRec",
        "NumOwned", "NumWant", "NumWish", "NumUserRatings", "MfgPlaytime"
    ]
    save_csv(table_descriptive_stats(df_clean, key_numeric), os.path.join(args.outdir, "table_3_2_descriptive_stats.csv"))

    # Table 3.1 (auto data dictionary for key vars + Cat/Rank + Rating)
    key_vars_for_dict = [
        "BGGId", "YearPublished", "GameWeight", "MinPlayers", "MaxPlayers",
        "NumOwned", "NumUserRatings", "MfgPlaytime", "ComAgeRec", "Family"
    ] + rank_cols + cat_cols + ["Rating"]
    save_csv(table_data_dictionary(df, key_vars_for_dict), os.path.join(args.outdir, "table_3_1_data_dictionary_auto.csv"))

    # Figures 3.1–3.6
    fig_year_distribution(df_clean, args.outdir)
    fig_gameweight_distribution(df_clean, args.outdir)
    fig_log1p_effect(df, args.outdir)                 # uses raw for before/after effect
    fig_gw_vs_nur_log1p(df_clean, args.outdir)        # uses NumUserRatings (before dropping)
    fig_corr_heatmap(df_clean, args.outdir)

    # Table 3.4 high correlation pairs (before hard drops, to match your narrative)
    pairs = high_corr_pairs(df_clean, threshold=args.corr_threshold)
    save_csv(pairs, os.path.join(args.outdir, "table_3_4_high_corr_pairs.csv"))

    # Drop non-model fields per your policy
    df_model = drop_non_model_fields(df_clean, drop_languageease=True)

    # Apply explicit redundancy elimination per your Chapter 3
    df_model = apply_redundancy_drops(df_model)
    save_csv(df_model, os.path.join(args.outdir, "DM1_prepared_after_drops.csv"))

    # Add log1p features (for later clustering/regression)
    df_logs = add_log1p_transforms(df_model)
    save_csv(df_logs, os.path.join(args.outdir, "DM1_prepared_with_logs.csv"))

    # Quick pipeline summary (for reporting)
    summary = pd.DataFrame([{
        "rows": df.shape[0],
        "cols_original": df.shape[1],
        "cols_after_clean_and_drops": df_model.shape[1],
        "num_cat_cols": len(cat_cols),
        "num_rank_cols": len(rank_cols),
        "rating_classes": ", ".join(map(str, sorted(df["Rating"].dropna().unique()))),
        "corr_threshold_used": args.corr_threshold
    }])
    save_csv(summary, os.path.join(args.outdir, "pipeline_summary.csv"))

    print("DONE. Outputs saved to:", args.outdir)


if __name__ == "__main__":
    main()
