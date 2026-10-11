"""End-to-end pipeline: data -> EDA -> clustering -> models (+LSTM) -> SHAP -> routing -> results.

Run from the project root:   python run_pipeline.py            (about 5-8 minutes on a laptop)
                              python run_pipeline.py --no-lstm  (skip the LSTM, ~2 minutes)
All figures go to outputs/figures, tables to outputs/tables, metrics to outputs/results.json,
and the files the dashboard reads go to outputs/app.
"""
import argparse
import json
import sys
import warnings

if sys.platform == "darwin":  # load PyTorch's OpenMP runtime before XGBoost's to avoid a macOS deadlock
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import clustering as CL
from src import eda, models as M, plots_routing as PR, routing as R
from src.config import CLASS_NAMES, OUT_DIR
from src.data import build_dataset, data_quality_report, split_masks

TAB = OUT_DIR / "tables"; TAB.mkdir(exist_ok=True)
APP = OUT_DIR / "app"; APP.mkdir(exist_ok=True)


def log(msg):
    print(f"\n=== {msg} ===", flush=True)


def main(use_lstm=True):
    results = {}
    log("1. Load + preprocess + label + features")
    raw, df, model_df, feats, _ = build_dataset()
    dq = data_quality_report(raw); dq.to_csv(TAB / "data_quality.csv", index=False)
    print(dq.to_string(index=False))
    results["data"] = {"rows_raw": int(len(raw)), "rows_model": int(len(model_df)),
                       "period": [str(raw.DateTime.min()), str(raw.DateTime.max())],
                       "junctions": int(raw.Junction.nunique())}

    log("2. EDA figures")
    eda.run_all(raw, df, model_df, feats)

    log("3. Clustering (congestion zones)")
    tr_full, _, _ = split_masks(df)
    cres = CL.run_clustering(df, tr_full, k=3)
    CL.plot_selection(cres); CL.plot_zones(cres); CL.plot_pca(cres)
    cres["selection"].round(4).to_csv(TAB / "cluster_selection.csv", index=False)
    cres["zone_summary"].to_csv(TAB / "zone_summary.csv")
    cres["zone_by_junction"].to_csv(TAB / "zone_by_junction.csv")
    cres["profiles"].to_csv(APP / "cluster_profiles.csv", index=False)
    print(cres["selection"].round(3).to_string(index=False)); print(cres["zone_summary"])
    results["clustering"] = {"kmeans": cres["kmeans"], "dbscan": cres["dbscan"], "pca_explained_var": cres["pca_var"],
                             "zone_summary": cres["zone_summary"].reset_index().to_dict("records"),
                             "zone_by_junction": cres["zone_by_junction"].reset_index().to_dict("records")}

    # zone of the target slot becomes a model feature (clustering -> prediction link)
    model_df = CL.zone_feature(model_df, cres["profiles"])
    feats = feats + ["hotspot_zone"]
    train, val, test = split_masks(model_df)
    results["split"] = {"train_rows": int(train.sum()), "val_rows": int(val.sum()), "test_rows": int(test.sum()),
                        "train": "2015-11-01 .. 2017-02-28", "val": "2017-03", "test": "2017-04-01 .. 2017-06-30",
                        "class_share_test": model_df.loc[test, "target_class"].value_counts(normalize=True)
                        .sort_index().round(3).rename(index=dict(enumerate(CLASS_NAMES))).to_dict()}
    results["features"] = feats

    log("4. Classification models (t+1 h congestion class)")
    cls, preds, models, info = M.train_classifiers(model_df, feats, train, val, test)
    results["xgb_info"] = info
    yte = model_df.loc[test, "target_class"].values

    if use_lstm:
        log("4b. LSTM")
        from src import lstm_model as L
        out = L.train_lstm(full_df=df, model_df=model_df, train=train, val=val, test=test)
        assert (out["test_index"] == model_df.index[test]).all()
        preds["LSTM"] = out["pred"]
        cls = pd.concat([cls, pd.DataFrame([M.clf_metrics("LSTM", out["y"], out["pred"], out["train_seconds"])])],
                        ignore_index=True)
        out["history"].to_csv(TAB / "lstm_history.csv", index=False)
        PR.plot_lstm_history(out["history"])
        M.plot_confusion(out["y"], out["pred"], "LSTM", "12b_confusion_lstm.png")
        lstm_proba = out["proba"]
    print(cls.round(3).to_string(index=False))
    cls.round(4).to_csv(TAB / "classification_metrics.csv", index=False)
    results["classification"] = cls.round(4).to_dict("records")
    M.plot_model_comparison(cls)
    M.plot_confusion(yte, preds["XGBoost"], "XGBoost", "12_confusion_xgboost.png")
    results["classification_report_xgb"] = M.classification_report(yte, preds["XGBoost"], target_names=CLASS_NAMES,
                                                                    output_dict=True)
    pj = M.per_junction_report(model_df, test, preds); pj.to_csv(TAB / "per_junction_xgb.csv")
    results["per_junction_xgb"] = pj.reset_index().to_dict("records")
    acc_h = M.per_hour_error(model_df, test, preds); M.plot_hour_error(acc_h)
    results["xgb_accuracy_by_hour"] = acc_h.round(3).to_dict()

    log("5. Regression variant (vehicles at t+1)")
    reg, rpreds = M.train_regressors(model_df, feats, train, val, test)
    print(reg.round(3).to_string(index=False))
    reg.round(4).to_csv(TAB / "regression_metrics.csv", index=False)
    results["regression"] = reg.round(4).to_dict("records")
    M.plot_regression(model_df, test, rpreds)

    log("6. SHAP explainability")
    imp = M.shap_analysis(models["XGBoost"], model_df, feats, test)
    imp.to_csv(TAB / "shap_importance.csv", index=False)
    print(imp.head(10).to_string(index=False))
    results["shap_top10"] = imp.head(10).to_dict("records")

    log("7. Route optimisation + evaluation")
    proba = models["XGBoost"].predict_proba(model_df.loc[test, feats])
    te = model_df.loc[test, ["Junction", "target_time", "target_class", "hist_vc_how", "target_vehicles"]].copy()
    te["hist_class"] = M.vc_to_class(te.hist_vc_how)
    te[["p_low", "p_moderate", "p_high"]] = proba
    te["xgb_pred"] = proba.argmax(1)
    if use_lstm:
        te[["lstm_p_low", "lstm_p_moderate", "lstm_p_high"]] = lstm_proba
    te.to_csv(APP / "test_states.csv", index=False)

    states = {}
    for t, g in te.groupby("target_time"):
        if g.Junction.nunique() < 4:
            continue
        states[t] = {"actual": {r.Junction: R.one_hot(r.target_class) for r in g.itertuples()},
                     "historical": {r.Junction: R.one_hot(r.hist_class) for r in g.itertuples()},
                     "predicted": {r.Junction: np.array([r.p_low, r.p_moderate, r.p_high]) for r in g.itertuples()}}
    trips = R.evaluate_routing(states, n_hours=300)
    rsum = R.summarise_routing(trips)
    th = pd.to_datetime(trips.time).dt.hour
    peak = th.isin([8, 9, 10, 17, 18, 19, 20])
    rsum["ml_peak_hours_mean_saving_%"] = round(float(trips.loc[peak, "ml_predicted_saving_%"].mean()), 2)
    rsum["ml_offpeak_mean_saving_%"] = round(float(trips.loc[~peak, "ml_predicted_saving_%"].mean()), 2)
    G = R.build_graph()
    rsum["graph"] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                     "total_km": round(sum(d["km"] for *_, d in G.edges(data=True)), 1)}
    print(json.dumps(rsum, indent=1))
    results["routing"] = rsum
    trips.groupby(["origin", "dest"])[["shortest_distance_min", "ml_predicted_min", "oracle_actual_min",
                                       "ml_predicted_saving_%"]].mean().round(2).to_csv(TAB / "routing_by_od.csv")
    PR.plot_savings(trips)
    # showcase example: a peak-hour trip where the ML route changed and saved the most
    # (a typical good case, not the single most extreme one: the trip at the 75th percentile of positive savings)
    cand = trips[peak & trips.ml_route_differs & (trips["ml_predicted_saving_%"] > 0)]
    cand = cand.sort_values("ml_predicted_saving_%").reset_index(drop=True)
    ex = cand.iloc[int(len(cand) * 0.75)]
    _, example = PR.plot_route_example(states[ex.time]["actual"], states[ex.time]["predicted"], ex.origin, ex.dest, ex.time)
    results["route_example"] = example

    # Dijkstra vs A*: identical optimal cost, compare runtime
    import time as _t
    some = list(states)[:50]; nodes = list(R.NODES)
    for algo in ("dijkstra", "astar"):
        t0 = _t.time(); costs = []
        for t in some:
            for a, b in zip(nodes, nodes[::-1]):
                if a != b:
                    costs.append(R.route(G, a, b, states[t]["predicted"], algo=algo)[1])
        results.setdefault("dijkstra_vs_astar", {})[algo] = {"queries": len(costs), "seconds": round(_t.time() - t0, 3),
                                                             "total_cost": round(float(np.sum(costs)), 3)}
    print(results["dijkstra_vs_astar"])

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log("Done - see outputs/results.json, outputs/figures, outputs/tables")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lstm", action="store_true")
    main(use_lstm=not ap.parse_args().no_lstm)
