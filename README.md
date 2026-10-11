# Traffic Congestion Prediction & Route Optimisation (ML-II, Project 3)

Predicts next-hour congestion (Low / Moderate / High) at road junctions, finds recurring congestion
hotspots with clustering, explains predictions with SHAP, and recommends congestion-aware routes on a
Bengaluru junction graph. It then measures whether those routes actually save time.

## Quick start (Mac)

```bash
cd ~/Desktop/traffic-congestion-ml
# recommended: a fresh environment (you have miniforge/conda installed)
conda create -n traffic python=3.11 -y
conda activate traffic
pip install -r requirements.txt

python run_pipeline.py        # ~5 min: regenerates every figure, table, model and outputs/results.json
streamlit run app.py          # opens the dashboard in your browser
jupyter notebook notebooks/Traffic_Congestion_Phase2.ipynb   # already executed, with all outputs
```

If XGBoost complains about `libomp.dylib`, run `conda install -c conda-forge llvm-openmp` (or `brew install libomp`).
To skip the LSTM for a faster run, use `python run_pipeline.py --no-lstm`.

## Project structure

```
data/traffic.csv                 Kaggle Traffic Prediction Dataset (fedesoriano): 4 junctions, hourly, Nov 2015 - Jun 2017
src/config.py                    paths, split dates, constants
src/data.py                      preprocessing, V/C congestion labels, feature engineering, chronological split
src/eda.py                       EDA figures
src/clustering.py                K-Means / DBSCAN congestion zones over (junction x hour-of-week)
src/models.py                    baselines, Logistic Regression, Random Forest, XGBoost, regression variant, SHAP
src/lstm_model.py                PyTorch LSTM (24 h sequence + static context)
src/routing.py                   Bengaluru junction graph, Dijkstra / A*, route evaluation
src/plots_routing.py             routing / LSTM figures
run_pipeline.py                  runs everything end to end
app.py                           Streamlit dashboard (7 tabs, interactive route planner on a map)
notebooks/...Phase2.ipynb        executed walkthrough notebook (review evidence)
outputs/figures/                 all figures (numbered in report order)
outputs/tables/                  metrics tables (CSV)
outputs/results.json             every number used in the reports
```

## Method in one paragraph

- **Congestion label:** volume-to-capacity, where capacity is the junction's 95th-percentile volume over the previous 28 days. V/C < 0.5 is Low, 0.5–0.8 is Moderate, ≥ 0.8 is High.
- **Features:** cyclic time, lags (1, 2, 3, 23, 167 h), rolling stats, the historical weekly profile, and the K-Means hotspot zone.
- **Split:** chronological. Train Nov 2015 – Feb 2017, validation Mar 2017, test Apr – Jun 2017.
- **Routing:** each Bengaluru node takes the traffic profile of one dataset junction by road role. Edge time = length / (free-flow speed × 1.0, 0.6 or 0.3 for Low, Moderate or High). Routes are planned with predicted congestion and scored with actual congestion.

## Key results (test set, Apr - Jun 2017)

| Model | Accuracy | Macro-F1 | Recall (High) |
|---|---|---|---|
| Persistence (naive) | 0.713 | 0.716 | 0.770 |
| Historical average | 0.709 | 0.713 | 0.622 |
| Logistic Regression | 0.774 | 0.777 | 0.773 |
| Random Forest | 0.789 | 0.792 | 0.796 |
| **XGBoost** | **0.790** | **0.793** | 0.770 |
| LSTM | 0.787 | 0.788 | **0.855** |

Routing over 126,000 simulated trips:
- The ML planner reroutes 16.3% of trips, saving **9.1%** travel time on those.
- It captures **73%** of the perfect-knowledge gain.
- A historical-average planner gains almost nothing.

## Limitations
- The dataset has only 4 anonymous junctions, with no weather or incident data.
- The road graph and the congestion-to-speed mapping are simulated.
- The 1-hour data resolution limits forecasts to an hourly horizon.
