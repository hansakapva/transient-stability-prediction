"""Use the saved random forest on new cases.

A case = for each of the 10 generator buses B30..B39, 4 voltage samples (p.u.) and
4 frequency samples (Hz).

Two CSV layouts are accepted by `read_cases`:
  1. Wide, one row per case, 80 columns named like the training data
     (B30_Vmag_1 ... B39_Vmag_4, B30_FreqHz_1 ... B39_FreqHz_4). Extra columns are kept.
  2. One case as a 10-row table, one row per generator:
        Bus,V1,V2,V3,V4,F1,F2,F3,F4
        B30,0.95,0.95,0.95,0.95,50.00,49.99,49.99,49.99
        ...
"""
import numpy as np
import pandas as pd

from .config import BUSES, MODEL_FILE, N_SAMPLES
from .model import load_model, p_unstable

_cache = {}


def _bundle(bundle=None):
    if bundle is not None:
        return bundle
    if "b" not in _cache:
        _cache["b"] = load_model(MODEL_FILE)
    return _cache["b"]


def case_to_row(vmag, freq, buses=BUSES):
    """Turn two (10 x 4) arrays (rows = buses B30..B39, columns = samples 1..4) into one wide row."""
    vmag, freq = np.asarray(vmag, dtype=float), np.asarray(freq, dtype=float)
    shape = (len(buses), N_SAMPLES)
    if vmag.shape != shape or freq.shape != shape:
        raise ValueError(f"vmag and freq must both have shape {shape} (buses x samples); "
                         f"got {vmag.shape} and {freq.shape}")
    row = {}
    for i, bus in enumerate(buses):
        for k in range(N_SAMPLES):
            row[f"{bus}_Vmag_{k + 1}"] = vmag[i, k]
    for i, bus in enumerate(buses):
        for k in range(N_SAMPLES):
            row[f"{bus}_FreqHz_{k + 1}"] = freq[i, k]
    return row


def read_cases(path):
    """Read a CSV in either supported layout and return one wide row per case."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "Bus" not in df.columns:
        return df
    vcols, fcols = [f"V{k}" for k in range(1, N_SAMPLES + 1)], [f"F{k}" for k in range(1, N_SAMPLES + 1)]
    missing = [c for c in vcols + fcols if c not in df.columns]
    if missing:
        raise ValueError(f"Per-generator layout needs columns Bus,{','.join(vcols + fcols)}; missing {missing}")
    bus = df["Bus"].astype(str).str.strip().str.upper()
    df.index = bus.where(bus.str.startswith("B"), "B" + bus)
    absent = [b for b in BUSES if b not in df.index]
    if absent:
        raise ValueError(f"Per-generator layout is missing rows for buses {absent}")
    df = df.loc[BUSES]
    return pd.DataFrame([case_to_row(df[vcols].to_numpy(), df[fcols].to_numpy())])


def predict_dataframe(cases, bundle=None):
    """Add Prediction, P_unstable and Outside_training_range columns to `cases`."""
    b = _bundle(bundle)
    feats = b["features"]
    missing = [c for c in feats if c not in cases.columns]
    if missing:
        raise ValueError(f"{len(missing)} required input columns missing, e.g. {missing[:5]}")
    X = cases[feats].apply(pd.to_numeric, errors="coerce")
    bad = X.isna().any(axis=1)
    if bad.any():
        raise ValueError(f"Empty or non-numeric inputs in rows {list(cases.index[bad][:10])}")
    model = b["model"]
    pred = model.predict(X)
    prob = p_unstable(model, X)
    lo, hi = pd.Series(b["train_min"])[feats], pd.Series(b["train_max"])[feats]
    outside = (X < lo) | (X > hi)
    out = cases.copy()
    out["Prediction"] = [b["class_names"][int(p)] for p in pred]
    out["P_unstable"] = np.round(prob, 4)
    out["Outside_training_range"] = [";".join(r.index[r.to_numpy()]) for _, r in outside.iterrows()]
    return out


def predict_case(vmag, freq, bundle=None):
    """Predict one case from two (10 x 4) arrays: rows B30..B39, columns samples 1..4.

    Returns a dict: prediction ("Stable"/"Unstable"), p_unstable (predicted probability of
    instability), and the inputs that fall outside the range seen in training (if any).
    """
    r = predict_dataframe(pd.DataFrame([case_to_row(vmag, freq)]), bundle).iloc[0]
    return {"prediction": r["Prediction"], "p_unstable": float(r["P_unstable"]),
            "outside_training_range": [c for c in r["Outside_training_range"].split(";") if c]}
