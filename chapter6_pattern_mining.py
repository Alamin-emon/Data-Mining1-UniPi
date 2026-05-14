#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chapter 6 — Pattern Mining (Frequent Itemsets + Association Rules)

What it produces (inside --out folder):
- patterns_minsup_<X>.csv                      (all frequent itemsets for each minsup)
- rules_minsup_<X>_minconf_<Y>.csv             (association rules, consequent size=1)
- summary_patterns_rules.csv                   (counts vs minsup/minconf)
- plot_patterns_vs_minsup.png                  (#patterns vs minsup)
- plot_rules_vs_minconf.png                    (#rules vs minconf for a chosen minsup)
- hist_rules_confidence.png, hist_rules_lift.png
- rule_based_rating_metrics.txt (if Rating column exists)

Design:
- Each row = one transaction (board game)
- Items = binary flags where value==1 + binned numeric features + optional Rating label item
- Frequent mining uses ECLAT (vertical tidsets) for speed and fewer dependencies
"""

import os
import math
import argparse
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_binary_columns(df: pd.DataFrame):
    """Binary numeric columns with values subset of {0,1}."""
    binary_cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals = set(pd.unique(s.dropna()))
            if vals.issubset({0, 1}):
                binary_cols.append(c)
    return binary_cols


def discretize_quantiles(df: pd.DataFrame, col: str, q: int = 4):
    """
    Discretize numeric column into quantile bins using qcut.
    Returns a Series of bin labels or None if not possible.
    """
    s = df[col]
    if not pd.api.types.is_numeric_dtype(s):
        return None
    if s.dropna().nunique() < 2:
        return None
    try:
        b = pd.qcut(s, q=q, duplicates="drop")
        # If duplicates dropped too much (e.g., only 1 bin), skip
        if b.cat.categories.size < 2:
            return None
        # Label bins as Q1..Qk
        k = b.cat.categories.size
        labels = [f"Q{i+1}" for i in range(k)]
        b = pd.qcut(s, q=k, labels=labels, duplicates="drop")
        return b.astype("object")
    except Exception:
        return None


def build_transactions(
    df: pd.DataFrame,
    rating_col: str = "Rating",
    quant_cols=None,
    qbins: int = 4,
    max_items_per_txn_guard: int = 60,
):
    """
    Build list of transactions (list of sets of strings).
    """
    if quant_cols is None:
        quant_cols = []

    binary_cols = detect_binary_columns(df)

    # Discretize selected quantitative columns
    binned_cols = {}
    for qc in quant_cols:
        if qc in df.columns:
            b = discretize_quantiles(df, qc, q=qbins)
            if b is not None:
                binned_cols[qc] = b

    txns = []
    for idx, row in df.iterrows():
        items = set()

        # Binary flags
        for bc in binary_cols:
            v = row.get(bc, 0)
            if v == 1:
                items.add(f"{bc}=1")

        # Binned numeric features
        for qc, bseries in binned_cols.items():
            label = bseries.iloc[idx]
            if pd.notna(label):
                items.add(f"{qc}={label}")

        # Rating as an item (optional but useful to mine rules about target)
        if rating_col in df.columns:
            r = row.get(rating_col)
            if pd.notna(r):
                items.add(f"{rating_col}={str(r)}")

        # Guard against extreme item explosion (rare, but safe)
        if len(items) > max_items_per_txn_guard:
            items = set(sorted(items)[:max_items_per_txn_guard])

        txns.append(items)

    return txns, binary_cols, list(binned_cols.keys())


def eclat_frequent_itemsets(transactions, min_sup: float, max_len: int = 3):
    """
    ECLAT frequent itemset mining (vertical tidsets).
    Returns dict: frozenset(items) -> support_count
    """
    n = len(transactions)
    min_count = int(math.ceil(min_sup * n))

    # Build item -> tidset
    item_tid = defaultdict(set)
    for tid, items in enumerate(transactions):
        for it in items:
            item_tid[it].add(tid)

    # Keep only frequent singletons
    items = sorted([it for it, tids in item_tid.items() if len(tids) >= min_count])
    item_tid = {it: item_tid[it] for it in items}

    freq = {}

    def dfs(prefix_items, prefix_tidset, start_index):
        # Try extending with each next item
        for j in range(start_index, len(items)):
            it = items[j]
            new_items = prefix_items + (it,)
            new_tidset = item_tid[it] if prefix_tidset is None else (prefix_tidset & item_tid[it])
            sup = len(new_tidset)
            if sup >= min_count:
                fs = frozenset(new_items)
                freq[fs] = sup
                if len(new_items) < max_len:
                    dfs(new_items, new_tidset, j + 1)

    dfs(tuple(), None, 0)
    return freq


def itemsets_to_df(freq_itemsets: dict, n_transactions: int):
    rows = []
    for fs, sup_cnt in freq_itemsets.items():
        rows.append({
            "itemset": " & ".join(sorted(fs)),
            "length": len(fs),
            "support_count": int(sup_cnt),
            "support": sup_cnt / n_transactions
        })
    out = pd.DataFrame(rows).sort_values(["support", "length"], ascending=[False, True])
    return out


def extract_rules(freq_itemsets: dict, n_transactions: int, min_conf: float):
    """
    Extract association rules with consequent size=1.
    Only uses frequent itemsets (downward closure ensures subsets exist).
    Returns DataFrame of rules.
    """
    sup = {fs: cnt / n_transactions for fs, cnt in freq_itemsets.items()}
    rows = []

    # Pre-index for faster subset lookup
    # (frozenset is hashable)
    for fs, s_fs in sup.items():
        if len(fs) < 2:
            continue

        for rhs in fs:
            lhs = frozenset(set(fs) - {rhs})
            if len(lhs) == 0:
                continue
            if lhs not in sup:
                continue
            rhs_fs = frozenset([rhs])
            if rhs_fs not in sup:
                continue

            conf = s_fs / sup[lhs]
            if conf < min_conf:
                continue
            lift = conf / sup[rhs_fs] if sup[rhs_fs] > 0 else np.nan

            rows.append({
                "lhs": " & ".join(sorted(lhs)),
                "rhs": rhs,
                "support": s_fs,
                "confidence": conf,
                "lift": lift,
                "lhs_support": sup[lhs],
                "rhs_support": sup[rhs_fs],
                "len_lhs": len(lhs)
            })

    df_rules = pd.DataFrame(rows)
    if not df_rules.empty:
        df_rules = df_rules.sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
    return df_rules


def plot_line(x, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_hist(values, title, xlabel, outpath, bins=30):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def rule_based_rating_prediction(transactions, rating_prefix="Rating=",
                                 rules_df=None, out_txt_path=None,
                                 test_size=0.2, seed=42):
    """
    Very simple exploitation: use rules with RHS like 'Rating=High' etc.
    Predict rating for a game if any rule antecedent matches its items.
    Score rule by confidence * lift; choose best score among matched rules.
    Evaluate with accuracy + macro-F1 (computed manually to avoid sklearn version issues).
    """
    if rules_df is None or rules_df.empty:
        return

    # Build true labels from transactions (if Rating=... item exists)
    y = []
    X = []
    for t in transactions:
        rating_items = [it for it in t if it.startswith(rating_prefix)]
        if not rating_items:
            y.append(None)
        else:
            y.append(rating_items[0].split("=", 1)[1])
        X.append(t)

    # Filter rows with known labels
    idxs = [i for i, lab in enumerate(y) if lab is not None]
    if len(idxs) < 50:
        return

    X = [X[i] for i in idxs]
    y = [y[i] for i in idxs]

    # Train-test split (simple stratified-ish: shuffle then split)
    rng = np.random.RandomState(seed)
    order = np.arange(len(y))
    rng.shuffle(order)
    split = int((1 - test_size) * len(order))
    tr_idx = order[:split]
    te_idx = order[split:]

    # Majority fallback from train
    y_train = [y[i] for i in tr_idx]
    majority = pd.Series(y_train).value_counts().idxmax()

    # Prepare rating rules
    rr = rules_df[rules_df["rhs"].astype(str).str.startswith(rating_prefix)].copy()
    if rr.empty:
        return

    # Parse antecedents into sets
    rr["lhs_set"] = rr["lhs"].apply(lambda s: set(map(str.strip, s.split("&"))))
    rr["score"] = rr["confidence"].astype(float) * rr["lift"].astype(float)

    # Predict
    y_pred = []
    y_true = []
    for i in te_idx:
        t = X[i]
        candidates = rr[rr["lhs_set"].apply(lambda lhs: lhs.issubset(t))]
        if candidates.empty:
            pred = majority
        else:
            best = candidates.sort_values(["score", "confidence", "lift"], ascending=False).iloc[0]
            pred = best["rhs"].split("=", 1)[1]
        y_pred.append(pred)
        y_true.append(y[i])

    # Metrics
    labels = sorted(set(y_true) | set(y_pred))
    label_to_i = {lab: k for k, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[label_to_i[yt], label_to_i[yp]] += 1

    acc = np.trace(cm) / np.sum(cm) if np.sum(cm) else 0.0

    # Macro-F1
    f1s = []
    for k, lab in enumerate(labels):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    if out_txt_path:
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write("Rule-based Rating Prediction (from association rules)\n")
            f.write("====================================================\n\n")
            f.write(f"Test size: {test_size}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Macro-F1: {macro_f1:.4f}\n\n")
            f.write("Confusion Matrix (rows=true, cols=pred)\n")
            f.write("Labels: " + ", ".join(labels) + "\n")
            f.write(str(cm) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to prepared CSV (e.g., DM1_game_dataset.csv)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--rating_col", default="Rating", help="Rating column name (default: Rating)")
    ap.add_argument("--max_len", type=int, default=3, help="Max frequent itemset length (default: 3)")
    ap.add_argument("--qbins", type=int, default=4, help="Quantile bins for numeric discretization (default: 4)")
    ap.add_argument("--min_sups", nargs="+", type=float, default=[0.01, 0.02, 0.03],
                    help="List of min_support values, e.g. 0.01 0.02 0.03")
    ap.add_argument("--min_confs", nargs="+", type=float, default=[0.30, 0.40, 0.50, 0.60],
                    help="List of min_confidence values, e.g. 0.3 0.4 0.5 0.6")
    ap.add_argument("--quant_cols", nargs="*", default=[
        "WeightAverage", "MfgPlaytime", "NumOwned", "YearPublished", "AvgRating", "NumUserRatings"
    ], help="Numeric columns to discretize (edit if your column names differ)")
    ap.add_argument("--rules_cap", type=int, default=5000, help="Max rules to save per setting (default: 5000)")
    args = ap.parse_args()

    out_root = args.out
    safe_mkdir(out_root)

    df = pd.read_csv(args.input)

    # Build transactions
    transactions, binary_cols, used_quant_cols = build_transactions(
        df,
        rating_col=args.rating_col,
        quant_cols=args.quant_cols,
        qbins=args.qbins
    )

    n = len(transactions)

    # Save basic config info
    with open(os.path.join(out_root, "config_used.txt"), "w", encoding="utf-8") as f:
        f.write("Pattern Mining — Configuration Used\n")
        f.write("=================================\n\n")
        f.write(f"Rows/Transactions: {n}\n")
        f.write(f"Rating column: {args.rating_col if args.rating_col in df.columns else 'NOT FOUND'}\n")
        f.write(f"Binary columns detected: {len(binary_cols)}\n")
        f.write("Binary columns: " + ", ".join(binary_cols) + "\n\n")
        f.write(f"Quant columns requested: {args.quant_cols}\n")
        f.write(f"Quant columns actually used (binned): {used_quant_cols}\n")
        f.write(f"Quantile bins: {args.qbins}\n")
        f.write(f"Max itemset length: {args.max_len}\n")
        f.write(f"Min supports: {args.min_sups}\n")
        f.write(f"Min confidences: {args.min_confs}\n")

    # Frequent patterns for each minsup
    summary_rows = []
    minsup_to_itemsets = {}
    for ms in args.min_sups:
        freq = eclat_frequent_itemsets(transactions, min_sup=ms, max_len=args.max_len)
        minsup_to_itemsets[ms] = freq

        df_pat = itemsets_to_df(freq, n)
        df_pat.to_csv(os.path.join(out_root, f"patterns_minsup_{ms:.3f}.csv"), index=False)

        summary_rows.append({"min_sup": ms, "n_patterns": int(df_pat.shape[0])})

    df_summary = pd.DataFrame(summary_rows).sort_values("min_sup")
    df_summary.to_csv(os.path.join(out_root, "summary_patterns_only.csv"), index=False)

    # Plot patterns vs minsup
    plot_line(
        x=df_summary["min_sup"].tolist(),
        y=df_summary["n_patterns"].tolist(),
        title="Frequent Patterns vs min_support",
        xlabel="min_support",
        ylabel="#frequent itemsets",
        outpath=os.path.join(out_root, "plot_patterns_vs_minsup.png")
    )

    # Association rules across min_conf for one chosen minsup (middle one if possible)
    ms_for_rules = sorted(args.min_sups)[len(args.min_sups)//2]
    freq_for_rules = minsup_to_itemsets[ms_for_rules]

    rules_count_rows = []
    rules_for_best_conf = None
    best_conf_for_hist = None

    for mc in args.min_confs:
        df_rules = extract_rules(freq_for_rules, n, min_conf=mc)

        # Cap for saving (avoid massive files)
        if not df_rules.empty and df_rules.shape[0] > args.rules_cap:
            df_rules_save = df_rules.head(args.rules_cap).copy()
        else:
            df_rules_save = df_rules

        df_rules_save.to_csv(
            os.path.join(out_root, f"rules_minsup_{ms_for_rules:.3f}_minconf_{mc:.2f}.csv"),
            index=False
        )

        rules_count_rows.append({
            "min_sup_for_rules": ms_for_rules,
            "min_conf": mc,
            "n_rules_total": int(df_rules.shape[0]),
            "n_rules_saved": int(df_rules_save.shape[0])
        })

        # Keep one ruleset for histograms (use a moderate confidence if exists)
        if best_conf_for_hist is None and mc >= 0.4:
            best_conf_for_hist = mc
            rules_for_best_conf = df_rules.copy()

    df_rules_counts = pd.DataFrame(rules_count_rows).sort_values("min_conf")

    # Plot rules vs minconf (at chosen minsup)
    plot_line(
        x=df_rules_counts["min_conf"].tolist(),
        y=df_rules_counts["n_rules_total"].tolist(),
        title=f"Association Rules vs min_confidence (min_sup={ms_for_rules:.3f})",
        xlabel="min_confidence",
        ylabel="#rules (total)",
        outpath=os.path.join(out_root, "plot_rules_vs_minconf.png")
    )

    # Histograms for confidence and lift (chosen confidence set)
    if rules_for_best_conf is not None and not rules_for_best_conf.empty:
        plot_hist(
            values=rules_for_best_conf["confidence"].astype(float).values,
            title=f"Histogram of Rule Confidence (min_sup={ms_for_rules:.3f}, min_conf={best_conf_for_hist:.2f})",
            xlabel="confidence",
            outpath=os.path.join(out_root, "hist_rules_confidence.png")
        )
        # Drop NaN/inf lift
        lift_vals = pd.to_numeric(rules_for_best_conf["lift"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not lift_vals.empty:
            plot_hist(
                values=lift_vals.values,
                title=f"Histogram of Rule Lift (min_sup={ms_for_rules:.3f}, min_conf={best_conf_for_hist:.2f})",
                xlabel="lift",
                outpath=os.path.join(out_root, "hist_rules_lift.png")
            )

        # Exploit rules: rule-based Rating prediction (if Rating exists)
        if args.rating_col in df.columns:
            rule_based_rating_prediction(
                transactions=transactions,
                rating_prefix=f"{args.rating_col}=",
                rules_df=rules_for_best_conf,
                out_txt_path=os.path.join(out_root, "rule_based_rating_metrics.txt"),
                test_size=0.2,
                seed=42
            )

    # Final combined summary
    df_final_summary = df_summary.copy()
    df_final_summary["min_sup_for_rules"] = ms_for_rules
    # Merge counts (wide format)
    for mc in sorted(args.min_confs):
        cnt = int(df_rules_counts[df_rules_counts["min_conf"] == mc]["n_rules_total"].iloc[0])
        df_final_summary[f"n_rules_minconf_{mc:.2f}"] = cnt

    df_final_summary.to_csv(os.path.join(out_root, "summary_patterns_rules.csv"), index=False)

    print("DONE. Pattern mining outputs saved to:", out_root)
    print("Main files:",
          "summary_patterns_rules.csv, plot_patterns_vs_minsup.png, plot_rules_vs_minconf.png, patterns_*.csv, rules_*.csv")


if __name__ == "__main__":
    main()
