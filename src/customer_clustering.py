"""
Customer segmentation with K-Means, K-Medoids (PAM-style), and DBSCAN.
Uses the existing customer RFM dataset; no separate dataset is required.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


FEATURES = ["Recency", "Frequency", "Monetary"]


def load_rfm(path: str) -> pd.DataFrame:
    """Load and validate the existing customer RFM CSV."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"RFM dataset is missing required columns: {missing}")
    df = df.copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    # Log transform reduces the effect of very large monetary/frequency values.
    df[["Frequency", "Monetary"]] = np.log1p(
        df[["Frequency", "Monetary"]].clip(lower=0)
    )
    return df


def k_medoids_pam(X: np.ndarray, n_clusters: int = 4, random_state: int = 42,
                  max_iter: int = 100):
    """
    Simple Partitioning Around Medoids (PAM) implementation.
    Returns labels and medoid row indices. Intended for moderate data sizes.
    """
    n = len(X)
    if n_clusters < 2 or n_clusters >= n:
        raise ValueError("n_clusters must be between 2 and number of rows - 1.")
    rng = np.random.default_rng(random_state)
    medoids = rng.choice(n, size=n_clusters, replace=False)
    distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    for _ in range(max_iter):
        labels = distances[:, medoids].argmin(axis=1)
        old_cost = distances[np.arange(n), medoids[labels]].sum()
        best_cost, best_medoids = old_cost, medoids.copy()
        # Search swaps; early exit once a better swap is found each pass.
        improved = False
        for slot in range(n_clusters):
            non_medoids = np.setdiff1d(np.arange(n), medoids, assume_unique=False)
            # Limit candidate swaps for speed on larger datasets.
            if len(non_medoids) > 300:
                non_medoids = rng.choice(non_medoids, 300, replace=False)
            for candidate in non_medoids:
                proposal = medoids.copy()
                proposal[slot] = candidate
                proposal_labels = distances[:, proposal].argmin(axis=1)
                cost = distances[np.arange(n), proposal[proposal_labels]].sum()
                if cost + 1e-9 < best_cost:
                    best_cost, best_medoids = cost, proposal
                    improved = True
                    break
            if improved:
                break
        medoids = best_medoids
        if not improved:
            break
    labels = distances[:, medoids].argmin(axis=1)
    return labels, medoids


def run_clustering(rfm_path: str, output_dir: str, n_clusters: int = 4,
                   dbscan_eps: float = 0.8, min_samples: int = 8,
                   max_customers: int = 1500):
    """Run clustering methods and save labels, a PCA plot, and summary CSV."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_rfm(rfm_path)
    # Keep runtime manageable for PAM and DBSCAN; deterministic sample.
    if len(df) > max_customers:
        df = df.sample(max_customers, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].to_numpy())

    labels_by_method = {
        "K-Means": KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X),
        "DBSCAN": DBSCAN(eps=dbscan_eps, min_samples=min_samples).fit_predict(X),
    }
    medoid_labels, medoid_indices = k_medoids_pam(X, n_clusters=n_clusters)
    labels_by_method["K-Medoids (PAM)"] = medoid_labels

    # PCA is used only to display 3-D RFM data in a 2-D plot.
    coords = PCA(n_components=2, random_state=42).fit_transform(X)
    output = df.copy()
    output["PCA1"], output["PCA2"] = coords[:, 0], coords[:, 1]
    summary_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, labels) in zip(axes, labels_by_method.items()):
        output[name] = labels
        unique = sorted(set(labels))
        for label in unique:
            mask = labels == label
            label_name = "Noise" if label == -1 else f"Cluster {label}"
            ax.scatter(coords[mask, 0], coords[mask, 1], s=16, alpha=0.7,
                       label=label_name)
        cluster_count = len(set(labels) - {-1})
        noise_count = int(np.sum(labels == -1))
        score = np.nan
        # Silhouette is defined only when there are 2+ non-singleton clusters.
        non_noise = labels != -1
        if len(set(labels[non_noise])) >= 2 and non_noise.sum() > len(set(labels[non_noise])):
            try:
                score = silhouette_score(X[non_noise], labels[non_noise])
            except ValueError:
                pass
        summary_rows.append({
            "Algorithm": name,
            "Clusters (excluding noise)": cluster_count,
            "Noise points": noise_count,
            "Silhouette Score (non-noise)": score,
            "Customers analysed": len(df),
        })
        ax.set_title(name)
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "customer_clustering_comparison.png")
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    labels_path = os.path.join(output_dir, "customer_cluster_labels.csv")
    output.to_csv(labels_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "customer_clustering_summary.csv")
    summary.to_csv(summary_path, index=False)
    return summary, plot_path, labels_path
