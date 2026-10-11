"""Data loading, preprocessing, congestion labelling and feature engineering.

Congestion definition (volume-to-capacity, V/C):
    capacity_t = 95th percentile of the junction's hourly volume over the PREVIOUS 28 days
    vc_ratio_t = vehicles_t / capacity_t
    Low       : vc_ratio <  0.50
    Moderate  : 0.50 <= vc_ratio < 0.80
    High      : vc_ratio >= 0.80
Using a rolling, past-only capacity removes the strong long-term growth trend at each junction
(so "High" means "busy for this junction right now", not "it is 2017") and never looks into the future.
"""
import numpy as np
import pandas as pd

from .config import (CAPACITY_MIN_PERIODS, CAPACITY_QUANTILE, CAPACITY_WINDOW_H, DATA_PATH,
                     TRAIN_END, VAL_END)

VC_BANDS = (0.50, 0.80)


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["DateTime"])
    return df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)


def data_quality_report(df):
    """Simple data-quality checks reported in the notebook / report."""
    rows = []
    for j, g in df.groupby("Junction"):
        full = pd.date_range(g.DateTime.min(), g.DateTime.max(), freq="h")
        q1, q3 = g.Vehicles.quantile([0.25, 0.75])
        upper = q3 + 3 * (q3 - q1)
        rows.append({
            "junction": j, "start": g.DateTime.min(), "end": g.DateTime.max(), "rows": len(g),
            "missing_hours": len(full) - len(g), "null_values": int(g.Vehicles.isna().sum()),
            "duplicates": int(g.DateTime.duplicated().sum()),
            "mean_vehicles": round(g.Vehicles.mean(), 2), "max_vehicles": int(g.Vehicles.max()),
            "extreme_spikes(>Q3+3IQR)": int((g.Vehicles > upper).sum()),
        })
    return pd.DataFrame(rows)


def preprocess(df):
    """Reindex to a complete hourly grid, interpolate any gaps, add a clipped copy for features."""
    out = []
    for j, g in df.groupby("Junction"):
        g = g.set_index("DateTime").asfreq("h")
        g["Junction"] = j
        g["was_missing"] = g["Vehicles"].isna().astype(int)
        g["Vehicles"] = g["Vehicles"].interpolate(limit=3).ffill().bfill()
        # Outlier handling for FEATURES only: clip extreme spikes to the junction's rolling 99.5th pct.
        # (The label keeps the raw value - a genuine spike is real congestion.)
        cap = g["Vehicles"].rolling(CAPACITY_WINDOW_H, min_periods=24).quantile(0.995).shift(1)
        g["veh_clipped"] = np.minimum(g["Vehicles"], cap.fillna(g["Vehicles"]))
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def add_labels(df):
    df = df.copy()
    grp = df.groupby("Junction")["Vehicles"]
    df["capacity"] = grp.transform(lambda s: s.shift(1).rolling(
        CAPACITY_WINDOW_H, min_periods=CAPACITY_MIN_PERIODS).quantile(CAPACITY_QUANTILE))
    df["vc_ratio"] = df["Vehicles"] / df["capacity"]
    df["congestion"] = np.select(
        [df.vc_ratio < VC_BANDS[0], df.vc_ratio < VC_BANDS[1]], [0, 1], default=2)
    df.loc[df.vc_ratio.isna(), "congestion"] = -1
    return df


def add_features(df):
    """Features available at hour t, used to predict hour t+1."""
    df = df.copy().sort_values(["Junction", "DateTime"])
    t = df["DateTime"]
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["is_peak"] = df.hour.isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
    df["hour_of_week"] = df.dow * 24 + df.hour
    # cyclical encodings of the TARGET hour (t+1), since that's what we forecast
    th = (t + pd.Timedelta(hours=1))
    df["t1_hour_sin"] = np.sin(2 * np.pi * th.dt.hour / 24)
    df["t1_hour_cos"] = np.cos(2 * np.pi * th.dt.hour / 24)
    df["t1_dow_sin"] = np.sin(2 * np.pi * th.dt.dayofweek / 7)
    df["t1_dow_cos"] = np.cos(2 * np.pi * th.dt.dayofweek / 7)
    df["t1_is_weekend"] = (th.dt.dayofweek >= 5).astype(int)
    df["t1_is_peak"] = th.dt.hour.isin([8, 9, 10, 17, 18, 19, 20]).astype(int)

    g = df.groupby("Junction")
    v = "veh_clipped"
    df["vc_t"] = df["veh_clipped"] / df["capacity"]
    for lag in (1, 2, 3):
        df[f"vc_lag{lag}"] = g["vc_t"].shift(lag)
    # same hour yesterday / last week relative to the TARGET hour t+1
    df["vc_lag23"] = g["vc_t"].shift(23)
    df["vc_lag167"] = g["vc_t"].shift(167)
    df["veh_t"] = df[v]
    df["veh_lag1"] = g[v].shift(1)
    df["veh_lag23"] = g[v].shift(23)
    df["veh_lag167"] = g[v].shift(167)
    df["vc_roll3_mean"] = g["vc_t"].transform(lambda s: s.rolling(3).mean())
    df["vc_roll24_mean"] = g["vc_t"].transform(lambda s: s.rolling(24).mean())
    df["vc_roll24_std"] = g["vc_t"].transform(lambda s: s.rolling(24).std())
    df["vc_diff1"] = df["vc_t"] - df["vc_lag1"]
    df["weekend_x_hour"] = df["t1_is_weekend"] * df["t1_hour_sin"]
    for j in sorted(df.Junction.unique()):
        df[f"junction_{j}"] = (df.Junction == j).astype(int)

    # targets (t+1)
    df["target_class"] = g["congestion"].shift(-1)
    df["target_vehicles"] = g["Vehicles"].shift(-1)
    df["target_vc"] = g["vc_ratio"].shift(-1)
    df["target_time"] = g["DateTime"].shift(-1)
    return df


def add_historical_profile(df, train_mask):
    """Junction x hour-of-week mean V/C computed on TRAINING rows only (no leakage)."""
    df = df.copy()
    th = df["target_time"]
    df["t1_how"] = th.dt.dayofweek * 24 + th.dt.hour
    prof = (df.loc[train_mask & df.vc_ratio.notna()]
            .groupby(["Junction", "hour_of_week"])["vc_ratio"].mean().rename("hist_vc_how"))
    df = df.merge(prof, left_on=["Junction", "t1_how"], right_index=True, how="left")
    df["hist_vc_how"] = df["hist_vc_how"].fillna(df.groupby("Junction")["hist_vc_how"].transform("mean"))
    return df, prof


BASE_FEATURES = [
    "t1_hour_sin", "t1_hour_cos", "t1_dow_sin", "t1_dow_cos", "t1_is_weekend", "t1_is_peak", "month",
    "vc_t", "vc_lag1", "vc_lag2", "vc_lag3", "vc_lag23", "vc_lag167",
    "veh_t", "veh_lag1", "veh_lag23", "veh_lag167", "capacity",
    "vc_roll3_mean", "vc_roll24_mean", "vc_roll24_std", "vc_diff1", "weekend_x_hour",
    "junction_1", "junction_2", "junction_3", "junction_4", "hist_vc_how",
]


def split_masks(df):
    tt = df["DateTime"]
    train = tt <= TRAIN_END
    val = (tt > TRAIN_END) & (tt <= VAL_END)
    test = tt > VAL_END
    return train, val, test


def build_dataset(path=DATA_PATH, extra_features=None):
    """Full pipeline -> modelling frame + feature list."""
    raw = load_raw(path)
    df = add_features(add_labels(preprocess(raw)))
    train, _, _ = split_masks(df)
    df, profile = add_historical_profile(df, train)
    feats = list(BASE_FEATURES) + list(extra_features or [])
    usable = df[feats + ["target_class", "target_vehicles"]].notna().all(axis=1) & (df.target_class >= 0)
    df["usable"] = usable.astype(int)
    model_df = df[usable].copy()
    model_df["target_class"] = model_df["target_class"].astype(int)
    return raw, df, model_df, feats, profile
