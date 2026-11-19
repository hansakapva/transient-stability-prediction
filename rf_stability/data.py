"""Loading, describing and splitting the IEEE 39-bus transient-stability dataset."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import DATA_FILE, FEATURES, FREQ_FEATURES, GROUP, LABEL, SEED, TEST_FRACTION, \
    UNSTABLE, VAL_FRACTION, VOLT_FEATURES


def load_dataset(path=DATA_FILE):
    """Read the CSV and check that every model input and the label are present."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing {len(missing)} required columns, e.g. {missing[:5]}")
    return df


def add_scenario_details(df):
    """Add readable scenario columns parsed from Contingency_ID and Operating_Condition.

    These are for analysis and plots only; the model uses just the 80 measurements.
      Fault_Kind     : Generator trip / Line trip / SLG / LLG / 3PHG / Load disconnection
      Fault_Time_s   : fault duration in seconds (line trips and ground faults)
      Load_Change_pct: load change of the operating condition, e.g. -4.0 ... +4.0
    """
    out = df.copy()
    cid = out[GROUP].astype(str)
    kind = pd.Series("Other", index=out.index)
    kind[cid.str.startswith("GenTrip")] = "Generator trip"
    kind[cid.str.startswith("LineTrip")] = "Line trip"
    kind[cid.str.startswith("LoadDisconnect")] = "Load disconnection"
    for k in ("SLG", "LLG", "3PHG"):
        kind[cid.str.startswith(k + "_")] = k
    out["Fault_Kind"] = kind
    t = cid.str.extract(r"Fault(\d+)p(\d+)s")
    out["Fault_Time_s"] = (t[0] + "." + t[1]).astype(float)
    if "Operating_Condition" in out.columns:
        out["Load_Change_pct"] = out["Operating_Condition"].astype(str).str.rstrip("%").astype(float)
    return out


def get_xy(df):
    """Model inputs (80 columns, fixed order) and the label."""
    return df[FEATURES], df[LABEL]


def extreme_rows(df, v_max=1.1, f_dev=5.0):
    """Rows with a physically impossible reading (voltage above v_max p.u. or |f - 50| > f_dev Hz)."""
    v = (df[VOLT_FEATURES] > v_max).any(axis=1)
    f = ((df[FREQ_FEATURES] - 50.0).abs() > f_dev).any(axis=1)
    return v | f


def _group_split(df, fraction, seed):
    """Hold out about `fraction` of the contingencies, keeping the unstable share similar."""
    n_splits = max(2, round(1 / fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, hold_idx = next(splitter.split(df, df[LABEL], df[GROUP]))
    return df.iloc[keep_idx], df.iloc[hold_idx]


def split_train_val_test(df, test_fraction=TEST_FRACTION, val_fraction=VAL_FRACTION, seed=SEED,
                         extremes_in_train=True):
    """Split into train / validation / test sets (about 70 / 15 / 15).

    The split is made by Contingency_ID: all 5 load levels of one contingency go to the
    same set. A random row split would put near-identical cases (same fault, slightly
    different load) in both training and test, and the test score would look better
    than the model really is on a new contingency.

    extremes_in_train: the few contingencies that contain physically impossible readings
    are always placed in the training set, so the model is trained WITH them and a later
    experiment that removes or caps them is compared on identical validation and test sets.
    """
    forced_ids = set(df.loc[extreme_rows(df), GROUP]) if extremes_in_train else set()
    pool = df[~df[GROUP].isin(forced_ids)]
    rest, test = _group_split(pool, test_fraction, seed)
    train, val = _group_split(rest, val_fraction / (1 - test_fraction), seed + 1)
    train = pd.concat([train, df[df[GROUP].isin(forced_ids)]])
    sets = {"train": train, "validation": val, "test": test}
    ids = {k: set(v[GROUP]) for k, v in sets.items()}
    assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]), \
        "a contingency appears in more than one set"
    assert sum(len(v) for v in sets.values()) == len(df)
    return train, val, test


def split_summary(train, val, test):
    """Rows, contingencies and class counts of each set."""
    rows = []
    for name, d in (("Train", train), ("Validation", val), ("Test", test)):
        n_unstable = int((d[LABEL] == UNSTABLE).sum())
        rows.append({"set": name, "rows": len(d), "contingencies": d[GROUP].nunique(),
                     "stable": len(d) - n_unstable, "unstable": n_unstable,
                     "unstable_pct": round(100 * n_unstable / len(d), 1)})
    out = pd.DataFrame(rows).set_index("set")
    out["share_of_rows_pct"] = (100 * out["rows"] / out["rows"].sum()).round(1)
    return out


def quality_report(df):
    """Basic data-quality counts."""
    num = df[FEATURES]
    return pd.Series({
        "rows": len(df),
        "columns": df.shape[1],
        "missing values": int(df.isna().sum().sum()),
        "infinite values": int(np.isinf(num.to_numpy()).sum()),
        "duplicate rows": int(df.drop(columns=[c for c in ["Case_ID"] if c in df]).duplicated().sum()),
        "rows with voltage > 1.1 p.u.": int((df[VOLT_FEATURES] > 1.1).any(axis=1).sum()),
        "rows with |f - 50| > 5 Hz": int(((df[FREQ_FEATURES] - 50).abs() > 5).any(axis=1).sum()),
        "rows with an impossible reading": int(extreme_rows(df).sum()),
    }, name="count")
