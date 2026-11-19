"""Random forest: builder, tuning by grouped cross-validation, scoring, saving and loading."""
from datetime import datetime

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from .config import BUSES, FEATURES, MODEL_FILE, N_SAMPLES, N_TREES, RF_GRID, SEED, STABLE, UNSTABLE


def build_rf(max_depth=None, min_samples_leaf=1, max_features="sqrt", class_weight="balanced_subsample",
             n_estimators=N_TREES, n_jobs=1):
    """Random forest with class weights for the ~18% unstable cases.

    No scaling is needed: trees split on thresholds, so units and extreme values do not
    distort them.
    """
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                  class_weight=class_weight, random_state=SEED, n_jobs=n_jobs)


def p_unstable(model, X):
    """Predicted probability of instability (weighted share of trees voting Unstable)."""
    return model.predict_proba(X)[:, list(model.classes_).index(UNSTABLE)]


def tune_rf_with_cv(X_train, y_train, groups, grid=RF_GRID, n_splits=5, seed=SEED, n_jobs=-1):
    """Choose the forest settings by grouped k-fold cross-validation inside the training set.

    Each fold holds out about 1/n_splits of the training contingencies (all their load
    levels together). Returns (results table sorted best-first, best parameter dict).
    """
    search = GridSearchCV(build_rf(), grid, scoring="balanced_accuracy", n_jobs=n_jobs,
                          cv=StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed))
    search.fit(X_train, y_train, groups=groups)
    r = search.cv_results_
    res = pd.DataFrame({
        "combo": range(len(r["params"])),
        "max_depth": [str(p["max_depth"]) for p in r["params"]],          # "None", "16", "8"
        "min_samples_leaf": [p["min_samples_leaf"] for p in r["params"]],
        "max_features": [str(p["max_features"]) for p in r["params"]],
        "class_weight": [p["class_weight"] for p in r["params"]],
        "cv_bal_acc": r["mean_test_score"],
        "cv_std": r["std_test_score"],
    })
    res = res.sort_values(["cv_bal_acc", "cv_std"], ascending=[False, True]).reset_index(drop=True)
    best = dict(r["params"][int(res.loc[0, "combo"])])
    return res.drop(columns="combo"), best


def save_model(model, params, metrics, train_df, path=MODEL_FILE, notes=""):
    """Save the fitted forest with what predict.py needs to use and sanity-check it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "params": params,
        "features": FEATURES,
        "buses": BUSES,
        "n_samples": N_SAMPLES,
        "class_names": {STABLE: "Stable", UNSTABLE: "Unstable"},
        "threshold": 0.5,
        "metrics": metrics,
        "train_min": train_df[FEATURES].min().to_dict(),
        "train_max": train_df[FEATURES].max().to_dict(),
        "n_training_rows": len(train_df),
        "sklearn_version": sklearn.__version__,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
    }, path)
    return path


def load_model(path=MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run train_random_forest.py first.")
    return joblib.load(path)
