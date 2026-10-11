"""Congestion-aware route optimisation on a simulated Bengaluru junction graph.

Graph: 21 well-known Bengaluru junctions (approximate real coordinates) joined by 33 major road links.
Road length = straight-line (haversine) distance x 1.3 circuity factor.

Simulation assumption (stated in the report): the dataset has 4 anonymous junctions, so every
Bengaluru node is assigned one of the 4 observed traffic PROFILES by its road role:
    J1 -> major Outer-Ring-Road / highway junctions (heaviest, strongest trend)
    J2 -> busy secondary arterials
    J3 -> central-business-district junctions (spiky, event-driven)
    J4 -> lower-volume peripheral / residential junctions
At hour h, a node's congestion = the observed (or predicted) congestion of its profile junction at h.

Edge travel time = length / (free_flow_speed x speed_factor), speed_factor by congestion class:
    Low 1.0, Moderate 0.6, High 0.3   (an edge uses the mean factor of its two end nodes)
Planning with the ML model uses the EXPECTED factor  sum_c P(class=c) * factor_c.
Every route is then SCORED with the ACTUAL congestion that occurred - so the comparison is fair.
"""
import math

import networkx as nx
import numpy as np
import pandas as pd

NODES = {
    # name: (lat, lon, profile_junction, road_role)
    "Silk Board":        (12.9177, 77.6238, 1, "ORR"),
    "Marathahalli":      (12.9569, 77.7011, 1, "ORR"),
    "KR Puram":          (13.0076, 77.6950, 1, "ORR"),
    "Hebbal":            (13.0358, 77.5970, 1, "ORR"),
    "Bellandur":         (12.9260, 77.6762, 1, "ORR"),
    "Tin Factory":       (12.9967, 77.6669, 1, "ORR"),
    "Agara":             (12.9237, 77.6493, 2, "Arterial"),
    "HSR Layout":        (12.9116, 77.6474, 2, "Arterial"),
    "Domlur":            (12.9610, 77.6387, 2, "Arterial"),
    "Indiranagar":       (12.9784, 77.6408, 2, "Arterial"),
    "Nagawara":          (13.0450, 77.6210, 2, "Arterial"),
    "Koramangala":       (12.9352, 77.6245, 2, "Arterial"),
    "Electronic City":   (12.8452, 77.6602, 2, "Arterial"),
    "MG Road":           (12.9730, 77.6170, 3, "CBD"),
    "Richmond Circle":   (12.9650, 77.5960, 3, "CBD"),
    "Majestic":          (12.9767, 77.5713, 3, "CBD"),
    "Mekhri Circle":     (13.0140, 77.5830, 3, "CBD"),
    "Jayanagar":         (12.9250, 77.5938, 3, "CBD"),
    "BTM Layout":        (12.9166, 77.6101, 4, "Local"),
    "Whitefield":        (12.9868, 77.7366, 4, "Local"),
    "CMRIT (AECS Layout)": (12.9667, 77.7118, 4, "Local"),
}

EDGES = [
    ("Silk Board", "HSR Layout", "ORR"), ("Silk Board", "BTM Layout", "ORR"), ("Silk Board", "Koramangala", "Arterial"),
    ("Silk Board", "Electronic City", "Highway"), ("HSR Layout", "Electronic City", "Arterial"),
    ("HSR Layout", "Agara", "ORR"), ("Agara", "Bellandur", "ORR"), ("Agara", "Koramangala", "Arterial"),
    ("Bellandur", "Marathahalli", "ORR"), ("Marathahalli", "CMRIT (AECS Layout)", "Arterial"),
    ("CMRIT (AECS Layout)", "Whitefield", "Arterial"), ("Marathahalli", "Domlur", "Arterial"),
    ("Marathahalli", "KR Puram", "ORR"), ("Whitefield", "KR Puram", "Arterial"), ("KR Puram", "Tin Factory", "Highway"),
    ("Tin Factory", "Indiranagar", "Arterial"), ("Tin Factory", "Nagawara", "ORR"), ("Nagawara", "Hebbal", "ORR"),
    ("Indiranagar", "Domlur", "Arterial"), ("Domlur", "Koramangala", "Arterial"), ("Domlur", "MG Road", "Arterial"),
    ("Indiranagar", "MG Road", "Arterial"), ("MG Road", "Richmond Circle", "Local"), ("Koramangala", "Richmond Circle", "Arterial"),
    ("Richmond Circle", "Jayanagar", "Arterial"), ("Jayanagar", "BTM Layout", "Arterial"), ("Richmond Circle", "Majestic", "Local"),
    ("Majestic", "Mekhri Circle", "Arterial"), ("Mekhri Circle", "Hebbal", "Highway"), ("MG Road", "Mekhri Circle", "Arterial"),
    ("Koramangala", "BTM Layout", "Local"), ("Bellandur", "Koramangala", "Arterial"), ("Hebbal", "Indiranagar", "Arterial"),
]

FREE_FLOW_KMPH = {"Highway": 45.0, "ORR": 40.0, "Arterial": 30.0, "Local": 25.0}
SPEED_FACTOR = np.array([1.0, 0.6, 0.3])  # Low, Moderate, High
CIRCUITY = 1.3


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def build_graph():
    G = nx.Graph()
    for n, (lat, lon, prof, role) in NODES.items():
        G.add_node(n, lat=lat, lon=lon, profile=prof, role=role)
    for u, v, road in EDGES:
        km = haversine_km(NODES[u], NODES[v]) * CIRCUITY
        G.add_edge(u, v, km=km, road=road, v_free=FREE_FLOW_KMPH[road])
    assert nx.is_connected(G)
    return G


def node_factors(state):
    """state: {junction: probability vector (3,)} -> {node: expected speed factor}."""
    return {n: float(np.dot(state[NODES[n][2]], SPEED_FACTOR)) for n in NODES}


def set_weights(G, state, attr):
    """Edge travel time in minutes under a congestion state (None -> free flow / distance-only)."""
    f = node_factors(state) if state is not None else None
    for u, v, d in G.edges(data=True):
        factor = 1.0 if f is None else (f[u] + f[v]) / 2
        d[attr] = 60 * d["km"] / (d["v_free"] * factor)


def path_cost(G, path, attr):
    return float(sum(G[u][v][attr] for u, v in zip(path[:-1], path[1:])))


def astar_heuristic_factory(G):
    vmax = max(FREE_FLOW_KMPH.values())

    def h(u, v):  # admissible: straight-line distance at the fastest possible speed (minutes)
        return 60 * haversine_km(NODES[u], NODES[v]) / vmax
    return h


def one_hot(c):
    v = np.zeros(3); v[int(c)] = 1.0
    return v


def route(G, src, dst, state, algo="dijkstra"):
    set_weights(G, state, "w")
    if algo == "astar":
        p = nx.astar_path(G, src, dst, heuristic=astar_heuristic_factory(G), weight="w")
    else:
        p = nx.dijkstra_path(G, src, dst, weight="w")
    return p, path_cost(G, p, "w")


def evaluate_routing(states_by_hour, n_hours=300, seed=42):
    """states_by_hour: {timestamp: {"actual": {j: onehot}, "predicted": {j: proba}, "historical": {j: onehot}}}
    Returns per-trip DataFrame. Every planner's route is scored with ACTUAL congestion."""
    G = build_graph()
    rng = np.random.default_rng(seed)
    hours = sorted(states_by_hour)
    if len(hours) > n_hours:
        hours = sorted(rng.choice(hours, n_hours, replace=False))
    nodes = list(NODES)
    rows = []
    for t in hours:
        st = states_by_hour[t]
        set_weights(G, st["actual"], "actual")
        plans = {}
        for name, s in (("shortest_distance", None), ("historical_avg", st["historical"]),
                        ("ml_predicted", st["predicted"]), ("oracle_actual", st["actual"])):
            set_weights(G, s, "plan")
            plans[name] = dict(nx.all_pairs_dijkstra_path(G, weight="plan"))
        for a in nodes:
            for b in nodes:
                if a == b:
                    continue
                r = {"time": t, "origin": a, "dest": b}
                for name, paths in plans.items():
                    r[f"{name}_min"] = path_cost(G, paths[a][b], "actual")
                r["ml_route_differs"] = plans["ml_predicted"][a][b] != plans["shortest_distance"][a][b]
                rows.append(r)
    trips = pd.DataFrame(rows)
    for name in ("historical_avg", "ml_predicted", "oracle_actual"):
        trips[f"{name}_saving_%"] = 100 * (trips.shortest_distance_min - trips[f"{name}_min"]) / trips.shortest_distance_min
    return trips


def summarise_routing(trips):
    out = {}
    for name in ("historical_avg", "ml_predicted", "oracle_actual"):
        s = trips[f"{name}_saving_%"]
        out[name] = {"mean_saving_%": round(float(s.mean()), 2), "median_saving_%": round(float(s.median()), 2),
                     "p90_saving_%": round(float(s.quantile(0.9)), 2),
                     "trips_faster_%": round(float((s > 0.01).mean() * 100), 1),
                     "trips_slower_%": round(float((s < -0.01).mean() * 100), 1),
                     "mean_minutes_saved": round(float((trips.shortest_distance_min - trips[f"{name}_min"]).mean()), 2)}
    d = trips[trips.ml_route_differs]
    out["ml_when_route_changes"] = {"share_of_trips_%": round(float(len(d) / len(trips) * 100), 1),
                                    "mean_saving_%": round(float(d["ml_predicted_saving_%"].mean()), 2)}
    gap = trips.shortest_distance_min - trips.oracle_actual_min
    got = trips.shortest_distance_min - trips.ml_predicted_min
    out["ml_share_of_oracle_gain_%"] = round(float(got.sum() / gap.sum() * 100), 1)
    out["n_trips"] = int(len(trips)); out["n_hours"] = int(trips.time.nunique())
    return out
