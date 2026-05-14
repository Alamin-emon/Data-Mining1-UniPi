"""
Chapter 4 — Clustering (BGG dataset)

What this script does:
1) Loads your prepared CSV (recommended: DM1_prepared_with_logs.csv).
2) Builds a clustering feature matrix:
   - Uses numeric columns + Cat:* binary flags
   - Excludes identifiers/text and supervised target columns (e.g., Rating)
   - Prefer *_log1p columns if present (drops corresponding raw count columns)
   - Excludes Rank:* by default (can include with --include-ranks)
3) Standardizes features (Z-score).
4) Runs:
   - K-Means (grid over k), selects best k (silhouette, tie-break DB index)
   - DBSCAN (grid over eps + min_samples), selects best config with constraints
   - Hierarchical (Ward): chooses best k via silhouette on full data
     + builds a dendrogram on a sample for report visualization
5) Saves:
   - CSVs: grid results, cluster sizes, cluster profiles, summary
   - PNGs: elbow, silhouette, PCA plots, DBSCAN k-distance plot(s), dendrogram

Install requirements (if missing):
  pip install pandas numpy matplotlib scikit-learn scipy
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from scipy.cluster.hierarchy import linkage, dendrogram


# -------------------------
# Utilities
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def detect_prefix_cols(cols, prefix: str):
    return [c for c in cols if c.startswith(prefix)]

def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def evaluate_clustering(Xs: np.ndarray, labels: np.ndarray):
    """
    Returns (silhouette, davies_bouldin, calinski_harabasz).
    Handles DBSCAN noise (-1) by evaluating on non-noise points when possible.
    """
    uniq = np.unique(labels)

    # If everything is one cluster or all noise
    if len(uniq) < 2 or (len(uniq) == 1 and uniq[0] == -1):
        return np.nan, np.nan, np.nan

    # If DBSCAN has noise and at least 2 real clusters
    if -1 in uniq:
        real = labels != -1
        real_clusters = set(labels[real])
        if len(real_clusters) < 2:
            return np.nan, np.nan, np.nan
        sil = silhouette_score(Xs[real], labels[real])
        db = davies_bouldin_score(Xs[real], labels[real])
        ch = calinski_harabasz_score(Xs[real], labels[real])
        return float(sil), float(db), float(ch)

    # Standard case
    sil = silhouette_score(Xs, labels)
    db = davies_bouldin_score(Xs, labels)
    ch = calinski_harabasz_score(Xs, labels)
    return float(sil), float(db), float(ch)


# -------------------------
# Feature selection (aligned to your Chapter 3 policy)
# -------------------------
def build_feature_columns(df: pd.DataFrame, include_ranks: bool = False):
    cols = df.columns.tolist()
    cat_cols = detect_prefix_cols(cols, "Cat:")
    rank_cols = detect_prefix_cols(cols, "Rank:")

    # Drop obvious non-features / supervised target
    exclude = set()
    for c in ["BGGId", "Name", "Description", "ImagePath", "Family", "GoodPlayers", "Rating"]:
        if c in df.columns:
            exclude.add(c)

    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Prefer *_log1p where they exist; drop corresponding raw count columns
    log_cols = [c for c in df.columns if c.endswith("_log1p")]
    raw_to_drop = set([c.replace("_log1p", "") for c in log_cols])

    features = []
    for c in num_cols:
        if c in exclude:
            continue
        if (c in raw_to_drop) and (f"{c}_log1p" in df.columns):
            continue
        features.append(c)

    # Ensure Cat:* included even if read as object in some environments
    for c in cat_cols:
        if c not in exclude and c not in features:
            features.append(c)

    # Handle ranks
    if not include_ranks and rank_cols:
        features = [c for c in features if c not in rank_cols]

    # Remove constant or near-empty columns
    cleaned = []
    for c in features:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 50:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        cleaned.append(c)

    return cleaned, cat_cols, rank_cols, log_cols


def make_X(df: pd.DataFrame, feature_cols):
    X = to_numeric_df(df[feature_cols]).copy()
    # Median imputation (consistent, robust default)
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    return X, Xs, scaler


# -------------------------
# Visualization helpers
# -------------------------
def pca_scatter(Xs: np.ndarray, labels: np.ndarray, outpath: str, title: str):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=6)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_fig(outpath)


def cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    s = pd.Series(labels, name="cluster")
    t = s.value_counts(dropna=False).rename_axis("cluster").reset_index(name="count")
    t["pct"] = (t["count"] / len(labels)) * 100
    return t


def cluster_profiles_mean(X: pd.DataFrame, labels: np.ndarray, top_features: int = 12) -> pd.DataFrame:
    """
    Produces a compact, report-friendly profile table:
    - Mean per cluster for all features
    - Also computes a simple "importance" by absolute z-diff from global mean.
    """
    tmp = X.copy()
    tmp["cluster"] = labels
    means = tmp.groupby("cluster").mean(numeric_only=True)

    global_mean = X.mean(numeric_only=True)
    global_std = X.std(numeric_only=True).replace(0, np.nan)

    # For each cluster, rank features by |(cluster_mean - global_mean) / global_std|
    rows = []
    for cl in means.index:
        diff_z = ((means.loc[cl] - global_mean) / global_std).abs().sort_values(ascending=False)
        top = diff_z.head(top_features).index.tolist()
        for f in top:
            rows.append({
                "cluster": cl,
                "feature": f,
                "cluster_mean": float(means.loc[cl, f]),
                "global_mean": float(global_mean[f]),
                "abs_z_diff": float(diff_z[f]) if pd.notna(diff_z[f]) else np.nan
            })

    return pd.DataFrame(rows).sort_values(["cluster", "abs_z_diff"], ascending=[True, False])


# -------------------------
# K-Means
# -------------------------
def run_kmeans(Xs: np.ndarray, outdir: str, kmin: int = 2, kmax: int = 12):
    rows = []
    for k in range(kmin, kmax + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(Xs)
        sil, db, ch = evaluate_clustering(Xs, labels)
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch
        })

    res = pd.DataFrame(rows)

    # Elbow
    plt.figure(figsize=(7, 4))
    plt.plot(res["k"], res["inertia"], marker="o")
    plt.title("K-Means — Elbow (Inertia vs k)")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    save_fig(os.path.join(outdir, "kmeans_elbow.png"))

    # Silhouette
    plt.figure(figsize=(7, 4))
    plt.plot(res["k"], res["silhouette"], marker="o")
    plt.title("K-Means — Silhouette vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    save_fig(os.path.join(outdir, "kmeans_silhouette.png"))

    # Choose best by silhouette, tie-break: lower DB
    cand = res.dropna(subset=["silhouette"]).copy()
    best = cand.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).head(1).iloc[0]
    best_k = int(best["k"])

    final = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    labels = final.fit_predict(Xs)
    sil, db, ch = evaluate_clustering(Xs, labels)

    return res, best_k, labels, (sil, db, ch)


# -------------------------
# DBSCAN
# -------------------------
def run_dbscan(Xs: np.ndarray, outdir: str,
               min_samples_list=(5, 10, 20),
               eps_grid=None):
    if eps_grid is None:
        eps_grid = np.linspace(0.3, 2.5, 12)

    rows = []

    # k-distance plots for guidance
    for ms in min_samples_list:
        nn = NearestNeighbors(n_neighbors=ms)
        nn.fit(Xs)
        dist, _ = nn.kneighbors(Xs)
        kth = np.sort(dist[:, -1])

        plt.figure(figsize=(7, 4))
        plt.plot(kth)
        plt.title(f"DBSCAN — k-distance plot (k=min_samples={ms})")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{ms}-NN distance")
        save_fig(os.path.join(outdir, f"dbscan_kdist_ms{ms}.png"))

        for eps in eps_grid:
            model = DBSCAN(eps=float(eps), min_samples=int(ms))
            labels = model.fit_predict(Xs)

            clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_frac = float(np.mean(labels == -1))

            sil, db, ch = (np.nan, np.nan, np.nan)
            if clusters >= 2:
                sil, db, ch = evaluate_clustering(Xs, labels)

            rows.append({
                "eps": float(eps),
                "min_samples": int(ms),
                "clusters": int(clusters),
                "noise_frac": noise_frac,
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch
            })

    res = pd.DataFrame(rows)

    # Selection rule (report-friendly and defensible):
    # Prefer 2–10 clusters, noise <= 0.40, maximize silhouette.
    cand = res[
        (res["clusters"].between(2, 10)) &
        (res["noise_frac"] <= 0.40)
    ].dropna(subset=["silhouette"])

    if len(cand) == 0:
        # Fallback: allow higher noise but still require >=2 clusters
        cand = res[(res["clusters"] >= 2) & (res["noise_frac"] <= 0.60)].dropna(subset=["silhouette"])

    if len(cand) == 0:
        # Last fallback: choose config with most clusters and lowest noise
        best = res.sort_values(["clusters", "noise_frac"], ascending=[False, True]).head(1).iloc[0]
    else:
        best = cand.sort_values(["silhouette", "noise_frac"], ascending=[False, True]).head(1).iloc[0]

    best_eps = float(best["eps"])
    best_ms = int(best["min_samples"])

    model = DBSCAN(eps=best_eps, min_samples=best_ms)
    labels = model.fit_predict(Xs)
    sil, db, ch = evaluate_clustering(Xs, labels)

    return res, (best_eps, best_ms), labels, (sil, db, ch)


# -------------------------
# Hierarchical (Ward)
# -------------------------
def run_hierarchical(Xs: np.ndarray, outdir: str, kmin: int = 2, kmax: int = 12,
                     dendro_sample: int = 1500):
    # Dendrogram on a sample (full dendrogram with 21,925 points is not practical)
    n = Xs.shape[0]
    rs = np.random.RandomState(42)
    idx = rs.choice(n, size=min(dendro_sample, n), replace=False)
    Xs_s = Xs[idx]

    # Ward linkage requires Euclidean; consistent with standardized Euclidean space
    Z = linkage(Xs_s, method="ward", metric="euclidean")

    plt.figure(figsize=(10, 4))
    dendrogram(Z, no_labels=True)
    plt.title(f"Hierarchical Dendrogram (Ward) — sample n={len(idx)}")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    save_fig(os.path.join(outdir, "hierarchical_dendrogram.png"))

    # Choose best k using AgglomerativeClustering on full data
    rows = []
    best_k = None
    best_row = None

    for k in range(kmin, kmax + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(Xs)
        sil, db, ch = evaluate_clustering(Xs, labels)
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})

    res = pd.DataFrame(rows)
    cand = res.dropna(subset=["silhouette"]).copy()
    best_row = cand.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).head(1).iloc[0]
    best_k = int(best_row["k"])

    final = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    labels = final.fit_predict(Xs)
    sil, db, ch = evaluate_clustering(Xs, labels)

    return res, best_k, labels, (sil, db, ch)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/DM1_prepared_with_logs.csv",
                    help="Path to prepared dataset (recommended: DM1_prepared_with_logs.csv)")
    ap.add_argument("--outdir", default="outputs_clustering", help="Output folder for clustering results")
    ap.add_argument("--include-ranks", action="store_true", help="Include Rank:* columns as clustering features (not recommended)")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--dendro-sample", type=int, default=1500, help="Sample size for dendrogram only")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    if not os.path.exists(args.input):
        # allow running from same folder with file name only
        alt = os.path.basename(args.input)
        if os.path.exists(alt):
            args.input = alt
        else:
            raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, low_memory=False)

    feature_cols, cat_cols, rank_cols, log_cols = build_feature_columns(df, include_ranks=args.include_ranks)
    X, Xs, _ = make_X(df, feature_cols)

    # ---------- K-MEANS ----------
    km_grid, km_best_k, km_labels, km_metrics = run_kmeans(Xs, args.outdir, args.kmin, args.kmax)
    save_csv(km_grid, os.path.join(args.outdir, "kmeans_grid_results.csv"))
    save_csv(cluster_sizes(km_labels), os.path.join(args.outdir, "kmeans_cluster_sizes.csv"))
    save_csv(cluster_profiles_mean(X, km_labels, top_features=12), os.path.join(args.outdir, "kmeans_cluster_profiles_top12.csv"))
    pca_scatter(Xs, km_labels, os.path.join(args.outdir, "kmeans_pca.png"),
                f"K-Means (k={km_best_k}) | sil={km_metrics[0]:.3f} DB={km_metrics[1]:.3f}")

    # ---------- DBSCAN ----------
    db_grid, (best_eps, best_ms), db_labels, db_metrics = run_dbscan(Xs, args.outdir)
    save_csv(db_grid, os.path.join(args.outdir, "dbscan_grid_results.csv"))
    save_csv(cluster_sizes(db_labels), os.path.join(args.outdir, "dbscan_cluster_sizes.csv"))
    save_csv(cluster_profiles_mean(X, db_labels, top_features=12), os.path.join(args.outdir, "dbscan_cluster_profiles_top12.csv"))
    pca_scatter(Xs, db_labels, os.path.join(args.outdir, "dbscan_pca.png"),
                f"DBSCAN (eps={best_eps:.2f}, min_samples={best_ms}) | sil={db_metrics[0]:.3f} noise={np.mean(db_labels==-1):.2f}")

    # ---------- HIERARCHICAL ----------
    h_grid, h_best_k, h_labels, h_metrics = run_hierarchical(Xs, args.outdir, args.kmin, args.kmax, args.dendro_sample)
    save_csv(h_grid, os.path.join(args.outdir, "hierarchical_k_grid_results.csv"))
    save_csv(cluster_sizes(h_labels), os.path.join(args.outdir, "hierarchical_cluster_sizes.csv"))
    save_csv(cluster_profiles_mean(X, h_labels, top_features=12), os.path.join(args.outdir, "hierarchical_cluster_profiles_top12.csv"))
    pca_scatter(Xs, h_labels, os.path.join(args.outdir, "hierarchical_pca.png"),
                f"Hierarchical Ward (k={h_best_k}) | sil={h_metrics[0]:.3f} DB={h_metrics[1]:.3f}")

    # ---------- SUMMARY ----------
    summary = pd.DataFrame([{
        "input_file": args.input,
        "n_rows": len(df),
        "n_features_used": len(feature_cols),
        "include_ranks": bool(args.include_ranks),
        "kmeans_best_k": km_best_k,
        "kmeans_silhouette": km_metrics[0],
        "kmeans_davies_bouldin": km_metrics[1],
        "kmeans_calinski_harabasz": km_metrics[2],
        "dbscan_best_eps": best_eps,
        "dbscan_best_min_samples": best_ms,
        "dbscan_clusters": int(len(set(db_labels)) - (1 if -1 in db_labels else 0)),
        "dbscan_noise_frac": float(np.mean(db_labels == -1)),
        "dbscan_silhouette": db_metrics[0],
        "hierarchical_best_k": h_best_k,
        "hierarchical_silhouette": h_metrics[0],
        "hierarchical_davies_bouldin": h_metrics[1],
        "hierarchical_calinski_harabasz": h_metrics[2],
    }])
    save_csv(summary, os.path.join(args.outdir, "clustering_summary.csv"))

    # Save datasets with labels (optional but useful)
    out_k = df.copy()
    out_k["cluster_kmeans"] = km_labels
    out_k.to_csv(os.path.join(args.outdir, "dataset_with_kmeans_labels.csv"), index=False)

    out_d = df.copy()
    out_d["cluster_dbscan"] = db_labels
    out_d.to_csv(os.path.join(args.outdir, "dataset_with_dbscan_labels.csv"), index=False)

    out_h = df.copy()
    out_h["cluster_hierarchical"] = h_labels
    out_h.to_csv(os.path.join(args.outdir, "dataset_with_hierarchical_labels.csv"), index=False)

    print("DONE. All clustering outputs saved to:", args.outdir)


if __name__ == "__main__":
    main()
