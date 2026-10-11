"""LSTM extension: sequence model for t+1 congestion class.

Input  = the last 24 hours of the junction's traffic (V/C ratio, clipped volume / capacity, time encodings)
       + static context of the target hour (hour/day encodings, historical profile, hotspot zone, junction).
Output = P(Low, Moderate, High) at t+1.
Trained on Nov 2015 - Feb 2017, early-stopped on Mar 2017, tested on Apr - Jun 2017 (same split as the tree models).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import MODEL_DIR, RANDOM_STATE

# macOS: PyTorch and XGBoost/scikit-learn each ship their own OpenMP runtime; with several threads the
# two can deadlock (the run "hangs" at the first LSTM epoch). One thread is plenty for this small model.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    torch.set_num_threads(1)

SEQ_LEN = 24
STEP_FEATS = ["vc_t", "step_hour_sin", "step_hour_cos", "step_dow_sin", "step_dow_cos"]
STATIC_FEATS = ["t1_hour_sin", "t1_hour_cos", "t1_dow_sin", "t1_dow_cos", "t1_is_weekend", "hist_vc_how",
                "hotspot_zone", "junction_1", "junction_2", "junction_3", "junction_4"]


class CongestionLSTM(nn.Module):
    def __init__(self, n_step, n_static, hidden=64, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(n_step, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_classes))

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], static], dim=1))


def make_sequences(full_df, model_df):
    """Build (seq, static, y) arrays aligned with model_df rows (same index order)."""
    f = full_df.copy()
    f["vc_t"] = (f["veh_clipped"] / f["capacity"]).clip(upper=3)
    f["step_hour_sin"] = np.sin(2 * np.pi * f.DateTime.dt.hour / 24)
    f["step_hour_cos"] = np.cos(2 * np.pi * f.DateTime.dt.hour / 24)
    f["step_dow_sin"] = np.sin(2 * np.pi * f.DateTime.dt.dayofweek / 7)
    f["step_dow_cos"] = np.cos(2 * np.pi * f.DateTime.dt.dayofweek / 7)
    seqs, keep = [], []
    pos = {}
    arrays = {}
    for j, g in f.groupby("Junction"):
        arrays[j] = g[STEP_FEATS].to_numpy(dtype=np.float32)
        pos.update({(j, t): i for i, t in enumerate(g.DateTime)})
    for idx, r in zip(model_df.index, model_df[["Junction", "DateTime"]].itertuples(index=False)):
        i = pos[(r.Junction, r.DateTime)]
        if i + 1 < SEQ_LEN:
            continue
        w = arrays[r.Junction][i + 1 - SEQ_LEN:i + 1]
        if np.isnan(w).any():
            continue
        seqs.append(w); keep.append(idx)
    keep = pd.Index(keep)
    static = model_df.loc[keep, STATIC_FEATS].to_numpy(dtype=np.float32)
    y = model_df.loc[keep, "target_class"].to_numpy(dtype=np.int64)
    return np.stack(seqs), static, y, keep


def train_lstm(full_df, model_df, train, val, test, epochs=25, batch=256, lr=2e-3, verbose=True):
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    if verbose:
        print("building 24-hour input sequences ...", flush=True)
    seq, static, y, keep = make_sequences(full_df, model_df)
    if verbose:
        print(f"{len(y)} sequences, training on {torch.get_num_threads()} thread(s)", flush=True)
    tr, va, te = (m.loc[keep].to_numpy() for m in (train, val, test))
    # standardise with training statistics only
    s_mu, s_sd = seq[tr].reshape(-1, seq.shape[2]).mean(0), seq[tr].reshape(-1, seq.shape[2]).std(0) + 1e-6
    st_mu, st_sd = static[tr].mean(0), static[tr].std(0) + 1e-6
    seq = (seq - s_mu) / s_sd; static = (static - st_mu) / st_sd
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    Xs, Xc, Y = T(seq), T(static.astype(np.float32)), T(y)

    model = CongestionLSTM(seq.shape[2], static.shape[1])
    counts = np.bincount(y[tr], minlength=3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    tr_idx = np.where(tr)[0]
    best, best_state, history, patience = 1e9, None, [], 0
    t0 = time.time()
    for ep in range(epochs):
        model.train(); np.random.shuffle(tr_idx); tot = 0
        for b in range(0, len(tr_idx), batch):
            ii = tr_idx[b:b + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xs[ii], Xc[ii]), Y[ii])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(ii)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xs[va], Xc[va]), Y[va]).item()
            vacc = (model(Xs[va], Xc[va]).argmax(1) == Y[va]).float().mean().item()
        sched.step(vl)
        history.append({"epoch": ep + 1, "train_loss": tot / len(tr_idx), "val_loss": vl, "val_acc": vacc})
        if verbose:
            print(f"epoch {ep+1:2d} train_loss={tot/len(tr_idx):.4f} val_loss={vl:.4f} val_acc={vacc:.3f}", flush=True)
        if vl < best - 1e-4:
            best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(Xs[te], Xc[te]), 1).numpy()
    torch.save({"state_dict": model.state_dict(), "s_mu": s_mu, "s_sd": s_sd, "st_mu": st_mu, "st_sd": st_sd},
               MODEL_DIR / "lstm.pt")
    return {"proba": proba, "pred": proba.argmax(1), "y": y[te], "test_index": keep[te],
            "history": pd.DataFrame(history), "train_seconds": time.time() - t0}
