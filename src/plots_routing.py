"""Figures for the route-optimisation evaluation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLASS_COLORS, CLASS_NAMES, FIG_DIR
from . import routing as R


def plot_savings(trips):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    d = trips[trips.ml_route_differs]
    axes[0].hist(d["ml_predicted_saving_%"], bins=40, color="#1f77b4", alpha=.85)
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set(title="ML route vs shortest-distance route\n(trips where the ML route differs)",
                xlabel="Travel-time saving (%) measured with ACTUAL congestion", ylabel="Trips")
    trips = trips.assign(hour=pd.to_datetime(trips.time).dt.hour)
    by = trips.groupby("hour")[["historical_avg_saving_%", "ml_predicted_saving_%", "oracle_actual_saving_%"]].mean()
    by.plot(ax=axes[1], marker="o", color=["#999", "#1f77b4", "#2e9d5b"])
    axes[1].set(title="Mean travel-time saving by hour of day", xlabel="Hour", ylabel="Mean saving (%)")
    axes[1].legend(["Historical-average planner", "ML-predicted planner", "Oracle (perfect knowledge)"])
    path = FIG_DIR / "18_routing_savings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_route_example(state_actual, state_pred, origin, dest, when):
    G = R.build_graph()
    p_short, _ = R.route(G, origin, dest, None)
    p_ml, _ = R.route(G, origin, dest, state_pred, algo="astar")
    R.set_weights(G, state_actual, "actual")
    t_short, t_ml = R.path_cost(G, p_short, "actual"), R.path_cost(G, p_ml, "actual")
    fig, ax = plt.subplots(figsize=(9, 9))
    for u, v in G.edges():
        ax.plot([R.NODES[u][1], R.NODES[v][1]], [R.NODES[u][0], R.NODES[v][0]], color="#ccc", lw=2, zorder=1)
    for path, col, lab, off in ((p_short, "#d1495b", f"Shortest distance: {t_short:.0f} min", 0.0008),
                                (p_ml, "#1f77b4", f"ML congestion-aware (A*): {t_ml:.0f} min", -0.0008)):
        xs = [R.NODES[n][1] + off for n in path]; ys = [R.NODES[n][0] + off for n in path]
        ax.plot(xs, ys, color=col, lw=4, label=lab, zorder=2, alpha=.85)
    for n, (lat, lon, prof, _) in R.NODES.items():
        c = CLASS_NAMES[int(np.argmax(state_actual[prof]))]
        ax.scatter(lon, lat, s=120, color=CLASS_COLORS[c], edgecolor="k", zorder=3)
        ax.annotate(n, (lon, lat), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(title=f"{origin} -> {dest}, {pd.Timestamp(when):%a %d %b %Y %H:00}\n(node colour = actual congestion)",
           xlabel="Longitude", ylabel="Latitude")
    ax.legend(loc="lower left")
    path = FIG_DIR / "17_route_example.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path, {"origin": origin, "dest": dest, "time": str(when), "shortest_path": p_short, "ml_path": p_ml,
                  "shortest_actual_min": round(t_short, 1), "ml_actual_min": round(t_ml, 1)}


def plot_lstm_history(hist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].plot(hist.epoch, hist.train_loss, label="train"); axes[0].plot(hist.epoch, hist.val_loss, label="validation")
    axes[0].set(title="LSTM loss", xlabel="Epoch"); axes[0].legend()
    axes[1].plot(hist.epoch, hist.val_acc, color="#2e9d5b"); axes[1].set(title="LSTM validation accuracy", xlabel="Epoch")
    path = FIG_DIR / "19_lstm_training.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path
