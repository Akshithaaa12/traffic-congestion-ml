"""Unsupervised congestion-zone (hotspot) detection.

Each point = one (junction, hour-of-week) slot, i.e. 4 x 168 = 672 points, described by its
TRAINING-period congestion behaviour. K-Means groups slots into congestion zones; DBSCAN is
used as a density-based comparison that also flags anomalous slots as noise.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import FIG_DIR, RANDOM_STATE

ZONE_NAMES = ["Free-flow zone", "Moderate zone", "Hotspot zone"]
ZONE_COLORS = ["#2e9d5b", "#f0a202", "#d1495b"]
PROFILE_FEATURES = ["mean_vc", "p90_vc", "std_vc", "p_high", "rel_volume"]


def build_profiles(df, train_mask):
    d = df[train_mask & (df.congestion >= 0)].copy()
    d["rel_volume"] = d.Vehicles / d.groupby("Junction").Vehicles.transform("mean")
    prof = d.groupby(["Junction", "hour_of_week"]).agg(
        mean_vc=("vc_ratio", "mean"), p90_vc=("vc_ratio", lambda s: s.quantile(0.9)),
        std_vc=("vc_ratio", "std"), p_high=("congestion", lambda s: (s == 2).mean()),
        rel_volume=("rel_volume", "mean"), n=("vc_ratio", "size")).reset_index()
    prof["dow"] = prof.hour_of_week // 24
    prof["hour"] = prof.hour_of_week % 24
    return prof


def kmeans_selection(X, ks=range(2, 9)):
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_),
                     "davies_bouldin": davies_bouldin_score(X, km.labels_),
                     "calinski_harabasz": calinski_harabasz_score(X, km.labels_)})
    return pd.DataFrame(rows)


def run_clustering(df, train_mask, k=3):
    prof = build_profiles(df, train_mask)
    scaler = StandardScaler()
    X = scaler.fit_transform(prof[PROFILE_FEATURES])
    sel = kmeans_selection(X)

    km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(X)
    # order clusters by mean congestion so 0=free-flow ... k-1=hotspot
    order = prof.assign(c=km.labels_).groupby("c").mean_vc.mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    prof["zone"] = pd.Series(km.labels_).map(remap).values
    prof["zone_name"] = prof.zone.map(dict(enumerate(ZONE_NAMES)))

    # DBSCAN: eps from the knee of the 5-NN distance curve
    nn = NearestNeighbors(n_neighbors=5).fit(X)
    kdist = np.sort(nn.kneighbors(X)[0][:, -1])
    # small grid around the knee: keep the eps with best silhouette, >=2 clusters and <=15% noise
    best = None
    for q in np.linspace(0.30, 0.95, 27):
        e = float(np.quantile(kdist, q))
        lab = DBSCAN(eps=e, min_samples=5).fit_predict(X)
        msk = lab >= 0
        if len(set(lab[msk])) < 2 or (~msk).mean() > 0.15:
            continue
        s = silhouette_score(X[msk], lab[msk])
        if best is None or s > best[1]:
            best = (e, s)
    eps = best[0] if best else float(np.quantile(kdist, 0.90))
    db = DBSCAN(eps=eps, min_samples=5).fit(X)
    prof["dbscan"] = db.labels_
    db_mask = db.labels_ >= 0
    n_db = len(set(db.labels_[db_mask]))
    db_stats = {"eps": round(eps, 3), "min_samples": 5, "n_clusters": int(n_db),
                "noise_points": int((~db_mask).sum()),
                "silhouette_excl_noise": (round(float(silhouette_score(X[db_mask], db.labels_[db_mask])), 3)
                                          if n_db > 1 else None)}

    zone_summary = prof.groupby("zone_name").agg(
        slots=("zone", "size"), mean_vc=("mean_vc", "mean"), p_high=("p_high", "mean"),
        rel_volume=("rel_volume", "mean")).reindex(ZONE_NAMES).round(3)
    zone_by_junction = pd.crosstab(prof.Junction, prof.zone_name)[ZONE_NAMES]
    km_stats = {"k": k, "silhouette": round(float(silhouette_score(X, prof.zone)), 3),
                "davies_bouldin": round(float(davies_bouldin_score(X, prof.zone)), 3)}
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    prof[["pc1", "pc2"]] = pca.transform(X)
    return {"profiles": prof, "selection": sel, "kmeans": km_stats, "dbscan": db_stats,
            "zone_summary": zone_summary, "zone_by_junction": zone_by_junction,
            "pca_var": pca.explained_variance_ratio_.round(3).tolist(), "kdist": kdist}


def plot_selection(res):
    sel = res["selection"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].plot(sel.k, sel.inertia, "o-"); axes[0].set(title="Elbow: K-Means inertia", xlabel="k")
    axes[1].plot(sel.k, sel.silhouette, "o-", color="#d1495b"); axes[1].set(title="Silhouette score (higher better)", xlabel="k")
    axes[2].plot(np.arange(len(res["kdist"])), res["kdist"], color="#555")
    axes[2].axhline(res["dbscan"]["eps"], ls="--", color="#d1495b", label=f"eps={res['dbscan']['eps']}")
    axes[2].set(title="DBSCAN: sorted 5-NN distance", xlabel="points", ylabel="distance"); axes[2].legend()
    path = FIG_DIR / "08_cluster_selection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_zones(res):
    prof = res["profiles"]
    fig, axes = plt.subplots(4, 1, figsize=(15, 6.5), sharex=True)
    cmap = matplotlib.colors.ListedColormap(ZONE_COLORS)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for ax, (j, g) in zip(axes, prof.groupby("Junction")):
        grid = g.pivot(index="dow", columns="hour", values="zone").reindex(range(7))
        ax.imshow(grid.values, aspect="auto", cmap=cmap, vmin=0, vmax=2)
        ax.set_yticks(range(7)); ax.set_yticklabels(days, fontsize=7)
        ax.set_ylabel(f"J{j}", rotation=0, labelpad=15, fontsize=11)
        ax.grid(False)
    axes[-1].set_xticks(range(24)); axes[-1].set_xlabel("Hour of day")
    handles = [matplotlib.patches.Patch(color=c, label=n) for c, n in zip(ZONE_COLORS, ZONE_NAMES)]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("K-Means congestion zones per junction x day x hour (training period)", y=1.06)
    path = FIG_DIR / "09_congestion_zones.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_pca(res):
    prof = res["profiles"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for z, (name, col) in enumerate(zip(ZONE_NAMES, ZONE_COLORS)):
        s = prof[prof.zone == z]
        axes[0].scatter(s.pc1, s.pc2, s=14, color=col, label=name, alpha=.8)
    axes[0].set(title=f"K-Means (k=3), silhouette={res['kmeans']['silhouette']}", xlabel="PC1", ylabel="PC2")
    axes[0].legend()
    noise = prof.dbscan < 0
    axes[1].scatter(prof.pc1[~noise], prof.pc2[~noise], c=prof.dbscan[~noise], cmap="tab10", s=14)
    axes[1].scatter(prof.pc1[noise], prof.pc2[noise], marker="x", color="k", s=20, label="noise (anomalous slots)")
    axes[1].set(title=f"DBSCAN: {res['dbscan']['n_clusters']} clusters, {res['dbscan']['noise_points']} noise",
                xlabel="PC1"); axes[1].legend()
    path = FIG_DIR / "10_cluster_pca.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def zone_feature(df, prof):
    """Attach the K-Means zone of the TARGET slot (junction, hour-of-week at t+1) as a feature."""
    z = prof.set_index(["Junction", "hour_of_week"])["zone"].rename("hotspot_zone")
    return df.merge(z, left_on=["Junction", "t1_how"], right_index=True, how="left")
