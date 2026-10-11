"""Streamlit dashboard - Traffic Congestion Prediction & Route Optimisation.

Run (from the project root, after `python run_pipeline.py`):   streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import routing as R
from src.config import CLASS_COLORS, CLASS_NAMES, DATA_PATH, FIG_DIR, OUT_DIR

st.set_page_config(page_title="Traffic Congestion ML", page_icon="🚦", layout="wide")

# --- compatibility with older/newer Streamlit & Plotly versions ---------------------------------
_V = tuple(int(x) for x in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}
Scattermap = getattr(go, "Scattermap", None) or go.Scattermapbox
MAP_KEY = "map" if hasattr(go, "Scattermap") else "mapbox"
APP = OUT_DIR / "app"
TAB = OUT_DIR / "tables"


@st.cache_data
def load():
    if not (OUT_DIR / "results.json").exists():
        return None
    res = json.loads((OUT_DIR / "results.json").read_text())
    raw = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])
    states = pd.read_csv(APP / "test_states.csv", parse_dates=["target_time"])
    prof = pd.read_csv(APP / "cluster_profiles.csv")
    tables = {p.stem: pd.read_csv(p) for p in TAB.glob("*.csv")}
    return res, raw, states, prof, tables


data = load()
if data is None:
    st.error("No results found. Run `python run_pipeline.py` first, then refresh this page.")
    st.stop()
res, raw, states, prof, tables = data
cls = pd.DataFrame(res["classification"])


def fig(name, caption=None):
    p = FIG_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, **WIDE)


st.title("🚦 Traffic Congestion Prediction & Route Optimisation")
st.caption("ML-II Project-Based Assignment · CMR Institute of Technology · Kaggle Traffic Prediction Dataset "
           "(4 junctions, hourly, Nov 2015 - Jun 2017) · simulated Bengaluru road graph")

tabs = st.tabs(["Overview", "EDA", "Hotspot clustering", "Prediction", "Route planner", "Explainability",
                "Evaluation & impact"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    best = cls.sort_values("f1_macro", ascending=False).iloc[0]
    pers = cls[cls.model == "Persistence (naive)"].iloc[0]
    rt = res["routing"]
    c = st.columns(4)
    c[0].metric("Best model (macro-F1)", best.model, f"{best.f1_macro:.3f}")
    c[1].metric("Test accuracy", f"{best.accuracy:.1%}", f"{(best.accuracy - pers.accuracy) * 100:+.1f} pts vs naive")
    c[2].metric("Avg travel-time saving (route changed)", f"{rt['ml_when_route_changes']['mean_saving_%']:.1f}%",
                f"{rt['ml_when_route_changes']['share_of_trips_%']:.0f}% of trips rerouted")
    c[3].metric("Share of perfect-knowledge gain captured", f"{rt['ml_share_of_oracle_gain_%']:.0f}%")
    st.markdown("""
**Pipeline:** Traffic data → preprocessing & feature engineering → congestion labels (volume/capacity) →
**K-Means / DBSCAN hotspot zones** → **t+1 h congestion prediction** (Baselines, Logistic Regression, Random Forest,
XGBoost, LSTM) → **SHAP** explanations → **congestion-aware routing** (Dijkstra / A\\*) → evaluation.

**Congestion label:** `V/C = vehicles / capacity`, where capacity = 95th percentile of that junction's volume over the
previous 28 days. Low < 0.50 ≤ Moderate < 0.80 ≤ High.

**Split (chronological, no shuffling):** train Nov 2015 - Feb 2017 · validation Mar 2017 · test Apr - Jun 2017.
""")
    st.dataframe(cls.style.format({c: "{:.3f}" for c in cls.columns if c not in ("model",)}, na_rep="-"),
                 **WIDE, hide_index=True)

# ---------------------------------------------------------------- EDA
with tabs[1]:
    j = st.selectbox("Junction", sorted(raw.Junction.unique()), key="eda_j")
    g = raw[raw.Junction == j]
    gran = st.radio("Resolution", ["Daily mean", "Hourly"], horizontal=True)
    s = g.set_index("DateTime").Vehicles
    s = s.resample("D").mean() if gran == "Daily mean" else s
    st.plotly_chart(px.line(s.reset_index(), x="DateTime", y="Vehicles", title=f"Junction {j} volume"),
                    **WIDE)
    a, b = st.columns(2)
    hp = g.assign(hour=g.DateTime.dt.hour, type=np.where(g.DateTime.dt.dayofweek >= 5, "Weekend", "Weekday"))
    hp = hp.groupby(["hour", "type"]).Vehicles.mean().reset_index()
    a.plotly_chart(px.line(hp, x="hour", y="Vehicles", color="type", markers=True, title="Hourly profile"),
                   **WIDE)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pv = g.assign(hour=g.DateTime.dt.hour, day=g.DateTime.dt.day_name()).pivot_table(
        index="day", columns="hour", values="Vehicles", aggfunc="mean").reindex(days)
    b.plotly_chart(px.imshow(pv, color_continuous_scale="YlOrRd", aspect="auto", title="Day x hour heatmap"),
                   **WIDE)
    st.subheader("Data quality")
    st.dataframe(tables.get("data_quality"), **WIDE, hide_index=True)
    with st.expander("Static EDA figures (used in the report)"):
        for n in ["01_daily_trend.png", "04_distribution_growth.png", "05_label_distribution.png",
                  "06_vc_label_example.png", "07_feature_correlation.png"]:
            fig(n)

# ---------------------------------------------------------------- Clustering
with tabs[2]:
    km, db = res["clustering"]["kmeans"], res["clustering"]["dbscan"]
    c = st.columns(4)
    c[0].metric("K-Means k", km["k"]); c[1].metric("Silhouette", km["silhouette"])
    c[2].metric("DBSCAN clusters", db["n_clusters"]); c[3].metric("DBSCAN noise slots", db["noise_points"])
    st.markdown("Each point is one **(junction, hour-of-week)** slot (4 × 168 = 672), described by its training-period "
                "mean V/C, 90th-percentile V/C, variability, probability of High congestion and relative volume.")
    jj = st.selectbox("Junction", [1, 2, 3, 4], key="cl_j")
    g = prof[prof.Junction == jj].pivot(index="dow", columns="hour", values="zone")
    g.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    st.plotly_chart(px.imshow(g, color_continuous_scale=[[0, "#2e9d5b"], [0.5, "#f0a202"], [1, "#d1495b"]],
                              zmin=0, zmax=2, aspect="auto", title=f"Junction {jj}: 0 = free-flow, 1 = moderate, 2 = hotspot"),
                    **WIDE)
    a, b = st.columns(2)
    a.dataframe(tables.get("zone_summary"), hide_index=True, **WIDE)
    b.dataframe(tables.get("cluster_selection"), hide_index=True, **WIDE)
    fig("10_cluster_pca.png"); fig("08_cluster_selection.png")

# ---------------------------------------------------------------- Prediction
with tabs[3]:
    st.plotly_chart(px.bar(cls.melt(id_vars="model", value_vars=["accuracy", "f1_macro", "recall_high"]),
                           x="model", y="value", color="variable", barmode="group", range_y=[0, 1],
                           title="Test-set metrics (Apr - Jun 2017)"), **WIDE)
    a, b = st.columns(2)
    with a: fig("12_confusion_xgboost.png")
    with b: fig("12b_confusion_lstm.png")
    st.subheader("Explore predictions")
    c1, c2 = st.columns(2)
    jj = c1.selectbox("Junction", [1, 2, 3, 4], key="pr_j")
    day = c2.date_input("Week starting", value=pd.Timestamp("2017-05-08").date(),
                        min_value=pd.Timestamp("2017-04-01").date(), max_value=pd.Timestamp("2017-06-23").date())
    d = states[(states.Junction == jj) & (states.target_time >= pd.Timestamp(day)) &
               (states.target_time < pd.Timestamp(day) + pd.Timedelta(days=7))]
    f = go.Figure()
    f.add_scatter(x=d.target_time, y=d.target_class, mode="lines", name="Actual class", line=dict(color="#333"))
    f.add_scatter(x=d.target_time, y=d.xgb_pred, mode="lines", name="XGBoost prediction", line=dict(color="#d1495b", dash="dot"))
    f.add_scatter(x=d.target_time, y=d.p_high * 2, mode="lines", name="P(High) × 2", line=dict(color="#1f77b4", width=1), opacity=.6)
    f.update_yaxes(tickvals=[0, 1, 2], ticktext=CLASS_NAMES)
    f.update_layout(title=f"Junction {jj}: actual vs predicted congestion (t+1 h)", height=380)
    st.plotly_chart(f, **WIDE)
    st.dataframe(tables.get("per_junction_xgb"), hide_index=True)
    st.subheader("Regression variant (vehicles at t+1)")
    st.dataframe(tables.get("regression_metrics"), hide_index=True)
    fig("14_regression_actual_vs_pred.png"); fig("13_error_by_hour.png"); fig("19_lstm_training.png")

# ---------------------------------------------------------------- Route planner
with tabs[4]:
    st.markdown("Pick a trip in the **test period**. The congestion-aware route is planned with the model's "
                "*predicted* congestion and then both routes are timed with the congestion that *actually* happened.")
    G = R.build_graph()
    nodes = list(R.NODES)
    c = st.columns(4)
    o = c[0].selectbox("Origin", nodes, index=nodes.index("BTM Layout"))
    dsts = [n for n in nodes if n != o]
    de = c[1].selectbox("Destination", dsts, index=dsts.index("CMRIT (AECS Layout)") if "CMRIT (AECS Layout)" in dsts else 0)
    dd = c[2].date_input("Date", value=pd.Timestamp("2017-05-08").date(),
                         min_value=pd.Timestamp("2017-04-01").date(), max_value=pd.Timestamp("2017-06-30").date(), key="rt_d")
    hh = c[3].slider("Hour", 0, 23, 18)
    algo = st.radio("Search algorithm", ["astar", "dijkstra"], horizontal=True,
                    format_func=lambda x: {"astar": "A*", "dijkstra": "Dijkstra"}[x])
    t = pd.Timestamp(dd) + pd.Timedelta(hours=hh)
    g = states[states.target_time == t]
    if g.Junction.nunique() < 4:
        st.warning("No predictions for all 4 profiles at this hour - pick another time.")
    else:
        actual = {r.Junction: R.one_hot(r.target_class) for r in g.itertuples()}
        pred = {r.Junction: np.array([r.p_low, r.p_moderate, r.p_high]) for r in g.itertuples()}
        p_short, _ = R.route(G, o, de, None, algo)
        p_ml, plan_ml = R.route(G, o, de, pred, algo)
        R.set_weights(G, actual, "actual")
        t_short, t_ml = R.path_cost(G, p_short, "actual"), R.path_cost(G, p_ml, "actual")
        km = lambda p: sum(G[u][v]["km"] for u, v in zip(p[:-1], p[1:]))
        m = st.columns(4)
        m[0].metric("Shortest-distance route", f"{t_short:.0f} min", f"{km(p_short):.1f} km", delta_color="off")
        m[1].metric("ML congestion-aware route", f"{t_ml:.0f} min", f"{km(p_ml):.1f} km", delta_color="off")
        m[2].metric("Time saved", f"{t_short - t_ml:.0f} min", f"{(t_short - t_ml) / t_short:.0%}")
        m[3].metric("ML-planned ETA", f"{plan_ml:.0f} min")

        f = go.Figure()
        for u, v in G.edges():
            f.add_trace(Scattermap(lat=[R.NODES[u][0], R.NODES[v][0]], lon=[R.NODES[u][1], R.NODES[v][1]],
                                         mode="lines", line=dict(width=2, color="#aaa"), hoverinfo="skip", showlegend=False))
        for p, col, name in ((p_short, "#d1495b", "Shortest distance"), (p_ml, "#1f77b4", "ML congestion-aware")):
            f.add_trace(Scattermap(lat=[R.NODES[n][0] for n in p], lon=[R.NODES[n][1] for n in p], mode="lines",
                                         line=dict(width=6 if col == "#1f77b4" else 9, color=col), name=name, opacity=.8))
        pc = {n: CLASS_NAMES[int(np.argmax(pred[R.NODES[n][2]]))] for n in nodes}
        f.add_trace(Scattermap(lat=[R.NODES[n][0] for n in nodes], lon=[R.NODES[n][1] for n in nodes],
                                     mode="markers+text", text=nodes, textposition="top right",
                                     marker=dict(size=13, color=[CLASS_COLORS[pc[n]] for n in nodes]),
                                     hovertext=[f"{n}: predicted {pc[n]} (profile J{R.NODES[n][2]})" for n in nodes],
                                     name="Junction (predicted congestion)"))
        f.update_layout(**{MAP_KEY: dict(style="open-street-map", center=dict(lat=12.96, lon=77.65), zoom=10.6)},
                        height=620, margin=dict(l=0, r=0, t=0, b=0), legend=dict(x=0, y=1))
        st.plotly_chart(f, **WIDE)
        st.write("**Shortest:** " + " → ".join(p_short)); st.write("**ML route:** " + " → ".join(p_ml))
        st.caption("Node colours = predicted congestion. Profiles: J1 ORR/highway, J2 arterial, J3 CBD, J4 local. "
                   "Speeds: free-flow 25-45 km/h by road type × factor 1.0 / 0.6 / 0.3 for Low / Moderate / High.")

# ---------------------------------------------------------------- Explainability
with tabs[5]:
    a, b = st.columns(2)
    with a: fig("15_shap_summary_high.png")
    with b: fig("16_shap_bar.png")
    st.dataframe(tables.get("shap_importance"), hide_index=True, **WIDE)
    st.markdown("**Reading:** current load (`vc_t`), the junction's usual load for that hour of the week "
                "(`hist_vc_how`), short rolling averages and the K-Means `hotspot_zone` dominate - recent state + "
                "recurring weekly pattern explain most of next-hour congestion.")

# ---------------------------------------------------------------- Evaluation
with tabs[6]:
    rt = res["routing"]
    rows = [{"planner": k, **v} for k, v in rt.items() if k in ("historical_avg", "ml_predicted", "oracle_actual")]
    st.subheader(f"Route evaluation: {rt['n_trips']:,} trips ({rt['n_hours']} test hours × 420 origin-destination pairs)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, **WIDE)
    fig("18_routing_savings.png"); fig("17_route_example.png")
    st.markdown("""
### Role in smart transportation
* **Commuters & logistics:** next-hour congestion forecasts let navigation and last-mile delivery planners reroute
  *before* a jam forms rather than after.
* **Traffic authorities:** hotspot zones show *where and when* congestion recurs → signal-timing plans, traffic police
  deployment, targeted road-capacity investment.
* **Explainability:** SHAP shows *why* a junction is predicted to jam, which builds trust for operational use.
* **Limits:** 4 anonymous junctions, no weather/incident data, simulated road graph and speed-congestion mapping.
  Live sensor/GPS feeds and a real OSM network are the natural next step.
""")
