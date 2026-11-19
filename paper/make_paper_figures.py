"""Figures for the IEEE conference paper: single-column width (3.45 in), vector PDF.

Uses the saved random forest (models/stability_rf.joblib) and the same data split; nothing is retrained.
Run from any folder:  py paper/make_paper_figures.py
"""
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PAPER_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from rf_stability import data, evaluate, model, plots  # noqa: E402
from rf_stability.config import LABEL, REPORT_DIR, RF_GRID, STABLE, UNSTABLE  # noqa: E402

FIG_DIR = PAPER_DIR / "figures"
COL = 3.45                                   # IEEE two-column layout: column width in inches
VAL_C, TEST_C = "#1baf7a", "#6250d6"         # validated pair (aqua always labelled)
ST_C, UN_C = plots.STABLE_COLOR, plots.UNSTABLE_COLOR


def style():
    plots.apply_style()
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.titlesize": 8, "axes.titlelocation": "center", "axes.titleweight": "bold",
        "axes.titlepad": 4, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42,
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def no_frame(ax):
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_fault_time(df):
    d = df.dropna(subset=["Fault_Time_s"])
    rows = ["SLG", "LLG", "3PHG", "Line trip"]
    pv = d.pivot_table(index="Fault_Kind", columns="Fault_Time_s", values=LABEL,
                       aggfunc=lambda s: 100 * (s == UNSTABLE).mean()).reindex(rows)
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    im = ax.imshow(pv.values, cmap=plots.ORANGE_RAMP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(pv.shape[1]), [f"{t:g} s" for t in pv.columns])
    ax.set_yticks(range(pv.shape[0]), rows)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=7, color="white" if v > 55 else plots.INK)
    ax.set_xlabel("fault clearing time")
    no_frame(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("unstable cases (%)", fontsize=7)
    cb.outline.set_visible(False)
    save(fig, "fig_fault_time")


def fig_traces(df):
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.55), sharex=True)
    x = np.arange(1, 5)
    for c, bus in enumerate(("B33", "B34")):
        for r, (tag, unit) in enumerate((("Vmag", "voltage (p.u.)"), ("FreqHz", "frequency (Hz)"))):
            ax = axes[r, c]
            cols = [f"{bus}_{tag}_{k}" for k in x]
            for cls, color in ((STABLE, ST_C), (UNSTABLE, UN_C)):
                v = df.loc[df[LABEL] == cls, cols]
                ax.fill_between(x, v.quantile(0.25), v.quantile(0.75), color=color, alpha=0.2, linewidth=0)
                ax.plot(x, v.median(), color=color, marker="o", markeredgecolor="white", markeredgewidth=0.8)
            ax.set_title(f"{bus} {unit.split(' ')[0]}")
            if c == 0:
                ax.set_ylabel(unit)
            if r == 1:
                ax.set_xlabel("sample")
            ax.set_xticks(x)
    fig.legend(handles=[Patch(color=ST_C, label="Stable"), Patch(color=UN_C, label="Unstable")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig_traces")


def fig_cv_grid(cv, best):
    feats = [str(f) for f in RF_GRID["max_features"]]
    depths = [str(d) for d in RF_GRID["max_depth"]]
    leaves = list(RF_GRID["min_samples_leaf"])
    sub = cv[cv["class_weight"] == best["class_weight"]]
    vmin, vmax = sub["cv_bal_acc"].min(), sub["cv_bal_acc"].max()
    fig, axes = plt.subplots(1, 3, figsize=(COL, 1.6), sharey=True, layout="constrained")
    for ax, mf in zip(axes, feats):
        pv = sub[sub["max_features"].astype(str) == mf].pivot_table(
            index="max_depth", columns="min_samples_leaf", values="cv_bal_acc")
        pv.index = pv.index.astype(str)
        pv = pv.reindex(index=depths, columns=leaves)
        im = ax.imshow(pv.values, cmap=plots.BLUE_RAMP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3), [str(v) for v in leaves])
        ax.set_yticks(range(3), ["none" if d == "None" else d for d in depths])
        for i in range(3):
            for j in range(3):
                v = pv.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=5.5,
                        color="white" if (v - vmin) / (vmax - vmin) > 0.55 else plots.INK)
        if str(best["max_features"]) == mf:
            i, j = depths.index(str(best["max_depth"])), leaves.index(best["min_samples_leaf"])
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=plots.INK, linewidth=1.4))
        ax.set_title(mf, fontsize=7)
        ax.set_xlabel("min_samples_leaf", fontsize=6.5)
        no_frame(ax)
    axes[0].set_ylabel("max_depth")
    fig.suptitle("max_features", fontsize=7, fontweight="bold")
    cb = fig.colorbar(im, ax=axes, fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    save(fig, "fig_cv_grid")


def fig_confusion(cms):
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.6))
    for ax, (name, cm) in zip(axes, cms.items()):
        cm = cm.to_numpy()
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=plots.BLUE_RAMP, vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1f}%", ha="center", va="center", fontsize=7,
                        color="white" if pct[i, j] > 55 else plots.INK)
        ax.set_xticks([0, 1], ["Unstable", "Stable"])
        ax.set_yticks([0, 1], ["Unstable", "Stable"] if ax is axes[0] else ["", ""])
        ax.set_xlabel("predicted")
        if ax is axes[0]:
            ax.set_ylabel("actual")
        ax.set_title(name)
        no_frame(ax)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_confusion")


def fig_roc(curves):
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    ax.plot([0, 1], [0, 1], color=plots.AXIS, linewidth=0.8)
    for name, (y, p) in curves.items():
        is_u = np.asarray(y) == UNSTABLE
        fpr, tpr, _ = roc_curve(is_u, p)
        ax.plot(fpr, tpr, color=VAL_C if name == "Validation" else TEST_C,
                label=f"{name} (AUC = {roc_auc_score(is_u, p):.3f})")
    ax.set_xlim(-0.005, 0.25)
    ax.set_ylim(0.6, 1.01)
    ax.set_xlabel("false positive rate (stable cases flagged)")
    ax.set_ylabel("true positive rate (unstable found)")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig_roc")


def fig_importance(perm, impurity, top=8):
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.3))
    for ax, imp, xlabel, title in (
            (axes[0], perm, "drop in test balanced accuracy when shuffled", "(a) Permutation importance (test set)"),
            (axes[1], impurity, "mean decrease in Gini impurity", "(b) Impurity importance (training)")):
        t = imp.head(top).iloc[::-1]
        ax.barh(range(len(t)), t["importance"], xerr=t["std"] if t["std"].any() else None, color=plots.METRIC_COLOR,
                height=0.65, edgecolor="white", linewidth=0.6,
                error_kw={"ecolor": plots.INK_2, "elinewidth": 0.6, "capsize": 0})
        ax.set_yticks(range(len(t)), t["feature_group"])
        ax.axvline(0, color=plots.AXIS, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig_importance")


def main():
    style()
    df = data.add_scenario_details(data.load_dataset())
    train, val, test = data.split_train_val_test(df)
    bundle = model.load_model()
    final, best = bundle["model"], bundle["params"]
    selection = model.build_rf(**best, n_jobs=-1).fit(*data.get_xy(train))
    val_pred, val_prob, _ = evaluate.evaluate(selection, val)
    test_pred, test_prob, _ = evaluate.evaluate(final, test)
    # keep "None" as text (pandas would read it as missing) and depths as "16", "8"
    cv = pd.read_csv(REPORT_DIR / "cv_grid.csv", dtype={"max_depth": str, "max_features": str},
                     keep_default_na=False)

    fig_fault_time(df)
    fig_traces(df)
    fig_cv_grid(cv, best)
    fig_confusion({"Validation": evaluate.confusion(val[LABEL], val_pred),
                   "Test": evaluate.confusion(test[LABEL], test_pred)})
    fig_roc({"Validation": (val[LABEL], val_prob), "Test": (test[LABEL], test_prob)})
    perm = evaluate.grouped_permutation_importance(final, test)
    impurity = evaluate.impurity_importance_by_group(final)
    fig_importance(perm, impurity)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
