"""Supervised congestion prediction (t+1 hour): baselines, LR, Random Forest, XGBoost + SHAP."""
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .config import CLASS_NAMES, FIG_DIR, MODEL_DIR, RANDOM_STATE
from .data import VC_BANDS


def vc_to_class(vc):
    vc = np.asarray(vc, dtype=float)
    return np.select([vc < VC_BANDS[0], vc < VC_BANDS[1]], [0, 1], default=2)


def clf_metrics(name, y, p, secs=None):
    return {"model": name, "accuracy": accuracy_score(y, p),
            "precision_macro": precision_score(y, p, average="macro", zero_division=0),
            "recall_macro": recall_score(y, p, average="macro", zero_division=0),
            "f1_macro": f1_score(y, p, average="macro"), "f1_weighted": f1_score(y, p, average="weighted"),
            "recall_high": recall_score(y, p, labels=[2], average="macro", zero_division=0),
            "train_seconds": secs}


def reg_metrics(name, y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"model": name, "RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": mean_absolute_error(y, p),
            "MAPE_%": float(np.mean(np.abs((y - p) / np.maximum(y, 1))) * 100)}


def train_classifiers(model_df, feats, train, val, test):
    Xtr, ytr = model_df.loc[train, feats], model_df.loc[train, "target_class"]
    Xva, yva = model_df.loc[val, feats], model_df.loc[val, "target_class"]
    Xte, yte = model_df.loc[test, feats], model_df.loc[test, "target_class"]
    trva = train | val
    rows, preds, models = [], {}, {}

    # Baseline 1: persistence (congestion at t+1 = congestion at t)
    p = model_df.loc[test, "congestion"].astype(int).values
    rows.append(clf_metrics("Persistence (naive)", yte, p)); preds["Persistence (naive)"] = p
    # Baseline 2: historical average for that junction x hour-of-week (training data only)
    p = vc_to_class(model_df.loc[test, "hist_vc_how"].values)
    rows.append(clf_metrics("Historical average", yte, p)); preds["Historical average"] = p

    # Logistic Regression
    t0 = time.time()
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    lr.fit(model_df.loc[trva, feats], model_df.loc[trva, "target_class"])
    p = lr.predict(Xte); rows.append(clf_metrics("Logistic Regression", yte, p, time.time() - t0))
    preds["Logistic Regression"] = p; models["Logistic Regression"] = lr

    # Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=400, max_depth=16, min_samples_leaf=3, n_jobs=-1,
                                class_weight="balanced_subsample", random_state=RANDOM_STATE)
    rf.fit(model_df.loc[trva, feats], model_df.loc[trva, "target_class"])
    p = rf.predict(Xte); rows.append(clf_metrics("Random Forest", yte, p, time.time() - t0))
    preds["Random Forest"] = p; models["Random Forest"] = rf

    # XGBoost: early-stop on the March validation month, then refit on train+val
    t0 = time.time()
    params = dict(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=3, reg_lambda=1.0, objective="multi:softprob", eval_metric="mlogloss",
                  random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
    xgb_es = XGBClassifier(early_stopping_rounds=100, **params).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    best_n = int(xgb_es.best_iteration) + 1
    params["n_estimators"] = best_n
    xgb = XGBClassifier(**params).fit(model_df.loc[trva, feats], model_df.loc[trva, "target_class"])
    p = xgb.predict(Xte); rows.append(clf_metrics("XGBoost", yte, p, time.time() - t0))
    preds["XGBoost"] = p; models["XGBoost"] = xgb

    for name, m in models.items():
        joblib.dump(m, MODEL_DIR / f"clf_{name.lower().replace(' ', '_')}.joblib")
    res = pd.DataFrame(rows)
    return res, preds, models, {"xgb_best_n_estimators": best_n}


def train_regressors(model_df, feats, train, val, test):
    trva = train | val
    yte = model_df.loc[test, "target_vehicles"].values
    rows, preds = [], {}
    p = model_df.loc[test, "Vehicles"].values
    rows.append(reg_metrics("Persistence (naive)", yte, p)); preds["Persistence (naive)"] = p
    rf = RandomForestRegressor(n_estimators=400, max_depth=18, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(model_df.loc[trva, feats], model_df.loc[trva, "target_vehicles"])
    p = rf.predict(model_df.loc[test, feats]); rows.append(reg_metrics("Random Forest", yte, p)); preds["Random Forest"] = p
    xgb = XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                       random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
    xgb.fit(model_df.loc[trva, feats], model_df.loc[trva, "target_vehicles"])
    p = xgb.predict(model_df.loc[test, feats]); rows.append(reg_metrics("XGBoost", yte, p)); preds["XGBoost"] = p
    joblib.dump(xgb, MODEL_DIR / "reg_xgboost.joblib")
    return pd.DataFrame(rows), preds


def per_junction_report(model_df, test, preds, model="XGBoost"):
    d = model_df.loc[test, ["Junction", "target_class"]].assign(pred=preds[model], pers=preds["Persistence (naive)"])
    out = d.groupby("Junction").apply(lambda g: pd.Series({
        "accuracy": accuracy_score(g.target_class, g.pred), "f1_macro": f1_score(g.target_class, g.pred, average="macro"),
        "persistence_f1_macro": f1_score(g.target_class, g.pers, average="macro")}))
    return out.round(3)


def per_hour_error(model_df, test, preds, model="XGBoost"):
    d = model_df.loc[test, ["target_time", "target_class"]].assign(pred=preds[model])
    d["hour"] = d.target_time.dt.hour
    return d.groupby("hour").apply(lambda g: accuracy_score(g.target_class, g.pred))


def plot_confusion(y, p, name, fname):
    cm = confusion_matrix(y, p, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set(title=f"{name}: normalised confusion matrix (test)", xlabel="Predicted", ylabel="Actual")
    path = FIG_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_model_comparison(res):
    d = res.set_index("model")[["accuracy", "f1_macro", "recall_high"]]
    fig, ax = plt.subplots(figsize=(11, 4))
    d.plot(kind="bar", ax=ax, color=["#1f77b4", "#2ca02c", "#d1495b"])
    ax.set(title="Test-set comparison (Apr-Jun 2017, t+1 h congestion class)", ylim=(0, 1), xlabel="")
    ax.tick_params(axis="x", rotation=20); ax.legend(["Accuracy", "Macro F1", "Recall (High)"], loc="lower right")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7)
    path = FIG_DIR / "11_model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_hour_error(acc_by_hour):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(acc_by_hour.index, acc_by_hour.values, color="#00798c")
    ax.set(title="XGBoost test accuracy by hour of day (temporal error analysis)", xlabel="Target hour", ylabel="Accuracy", ylim=(0, 1))
    path = FIG_DIR / "13_error_by_hour.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def plot_regression(model_df, test, rpreds):
    d = model_df.loc[test, ["Junction", "target_time", "target_vehicles"]].assign(pred=rpreds["XGBoost"])
    d = d[(d.Junction == 1) & (d.target_time >= "2017-06-01") & (d.target_time < "2017-06-15")]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.plot(d.target_time, d.target_vehicles, color="#333", lw=1.2, label="Actual")
    ax.plot(d.target_time, d.pred, color="#d1495b", lw=1.2, ls="--", label="XGBoost (t+1 h)")
    ax.set(title="Junction 1, 1-14 Jun 2017: actual vs predicted vehicles (regression variant)", ylabel="Vehicles / hour")
    ax.legend()
    path = FIG_DIR / "14_regression_actual_vs_pred.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def shap_analysis(xgb, model_df, feats, test, n=3000):
    import shap
    X = model_df.loc[test, feats].sample(min(n, int(test.sum())), random_state=RANDOM_STATE)
    expl = shap.TreeExplainer(xgb)
    sv = expl.shap_values(X)
    sv = np.asarray(sv)
    # normalise to (n, features, classes)
    if sv.ndim == 3 and sv.shape[0] == len(CLASS_NAMES):
        sv = np.transpose(sv, (1, 2, 0))
    high = sv[:, :, 2]
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap_all": np.abs(sv).mean(axis=(0, 2)),
                        "mean_abs_shap_high": np.abs(high).mean(axis=0)}).sort_values("mean_abs_shap_all", ascending=False)
    imp["share_%"] = 100 * imp.mean_abs_shap_all / imp.mean_abs_shap_all.sum()

    plt.figure()
    shap.summary_plot(high, X, show=False, max_display=15)
    plt.title("SHAP: drivers of the 'High congestion' prediction (XGBoost)")
    p1 = FIG_DIR / "15_shap_summary_high.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close("all")

    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.feature, top.mean_abs_shap_all, color="#1f77b4")
    ax.set(title="Global feature importance (mean |SHAP|, all classes)", xlabel="mean |SHAP value|")
    p2 = FIG_DIR / "16_shap_bar.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)
    return imp.round(4)
