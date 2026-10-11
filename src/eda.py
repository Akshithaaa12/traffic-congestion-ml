"""Exploratory data analysis figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import CLASS_COLORS, CLASS_NAMES, FIG_DIR

sns.set_theme(style="whitegrid", context="notebook")
JPAL = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#9467bd"}


def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_timeseries(raw):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for j, g in raw.groupby("Junction"):
        daily = g.set_index("DateTime")["Vehicles"].resample("D").mean()
        ax.plot(daily.index, daily.values, label=f"Junction {j}", color=JPAL[j], lw=1.2)
    ax.set(title="Daily mean hourly vehicle count per junction (Nov 2015 - Jun 2017)",
           xlabel="Date", ylabel="Vehicles / hour")
    ax.legend()
    return _save(fig, "01_daily_trend.png")


def plot_hourly_profile(raw):
    d = raw.assign(hour=raw.DateTime.dt.hour, weekend=np.where(raw.DateTime.dt.dayofweek >= 5, "Weekend", "Weekday"))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True)
    for ax, (j, g) in zip(axes, d.groupby("Junction")):
        sns.lineplot(data=g, x="hour", y="Vehicles", hue="weekend", ax=ax, errorbar=None,
                     palette={"Weekday": "#d1495b", "Weekend": "#00798c"})
        ax.set(title=f"Junction {j}", xlabel="Hour of day", ylabel="Mean vehicles" if j == 1 else "")
        ax.legend(fontsize=8, title=None)
    fig.suptitle("Average hourly traffic profile: weekday vs weekend", y=1.03)
    return _save(fig, "02_hourly_profile.png")


def plot_heatmap(raw):
    d = raw[raw.Junction == 1].assign(hour=raw.DateTime.dt.hour, day=raw.DateTime.dt.day_name().str[:3])
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pv = d.pivot_table(index="day", columns="hour", values="Vehicles", aggfunc="mean").loc[order]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    sns.heatmap(pv, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean vehicles"})
    ax.set(title="Junction 1: mean volume by day-of-week x hour", xlabel="Hour of day", ylabel="")
    return _save(fig, "03_heatmap_dow_hour_j1.png")


def plot_distributions(raw):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.boxplot(data=raw, x="Junction", y="Vehicles", ax=axes[0], palette=JPAL, hue="Junction", legend=False)
    axes[0].set(title="Volume distribution per junction (note outlier spikes at J3)")
    m = raw.assign(month=raw.DateTime.dt.to_period("M").dt.to_timestamp())
    mm = m.groupby(["month", "Junction"]).Vehicles.mean().reset_index()
    sns.lineplot(data=mm, x="month", y="Vehicles", hue="Junction", palette=JPAL, marker="o", ax=axes[1])
    axes[1].set(title="Monthly mean volume (growth trend)", xlabel="", ylabel="Mean vehicles / hour")
    axes[1].tick_params(axis="x", rotation=30)
    return _save(fig, "04_distribution_growth.png")


def plot_label_distribution(df):
    d = df[df.congestion >= 0].copy()
    d["Congestion"] = d.congestion.map(dict(enumerate(CLASS_NAMES)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ct = pd.crosstab(d.Junction, d.Congestion, normalize="index")[CLASS_NAMES]
    ct.plot(kind="bar", stacked=True, ax=axes[0], color=[CLASS_COLORS[c] for c in CLASS_NAMES])
    axes[0].set(title="Congestion class share per junction", ylabel="Share", xlabel="Junction")
    axes[0].legend(title=None, loc="lower right")
    axes[0].tick_params(axis="x", rotation=0)
    hp = d.assign(hour=d.DateTime.dt.hour).groupby("hour").congestion.apply(lambda s: (s == 2).mean())
    axes[1].bar(hp.index, hp.values, color="#d1495b")
    axes[1].set(title="Share of hours labelled High congestion, by hour of day", xlabel="Hour", ylabel="P(High)")
    return _save(fig, "05_label_distribution.png")


def plot_vc_example(df):
    d = df[(df.Junction == 1) & (df.DateTime >= "2017-05-01") & (df.DateTime < "2017-05-15")]
    fig, ax1 = plt.subplots(figsize=(13, 4))
    ax1.plot(d.DateTime, d.Vehicles, color="#555", lw=1, label="Vehicles")
    ax1.plot(d.DateTime, d.capacity, color="#1f77b4", ls="--", lw=1.2, label="Capacity proxy (rolling P95)")
    for c, name in enumerate(CLASS_NAMES):
        s = d[d.congestion == c]
        ax1.scatter(s.DateTime, s.Vehicles, s=10, color=CLASS_COLORS[name], label=name, zorder=3)
    ax1.set(title="Junction 1, 1-14 May 2017: volume, capacity proxy and congestion label", ylabel="Vehicles / hour")
    ax1.legend(ncol=5, fontsize=8)
    return _save(fig, "06_vc_label_example.png")


def plot_correlation(model_df, feats):
    cols = [c for c in feats if not c.startswith("junction_")] + ["target_vc"]
    corr = model_df[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": .7})
    ax.set(title="Feature correlation matrix (incl. target V/C at t+1)")
    return _save(fig, "07_feature_correlation.png")


def run_all(raw, df, model_df, feats):
    return [plot_timeseries(raw), plot_hourly_profile(raw), plot_heatmap(raw), plot_distributions(raw),
            plot_label_distribution(df), plot_vc_example(df), plot_correlation(model_df, feats)]
