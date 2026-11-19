"""Project charts (matplotlib).

Colour rules used in every chart:
  Stable   = blue   #2a78d6      Unstable = orange #eb6834   (checked for colour-blind safety)
  Model results (no class meaning) = violet #6250d6, second series = aqua #1baf7a
  Heatmaps of one quantity = single-hue ramp (blue, or orange for "unstable %");
  correlation = blue <-> red with a grey midpoint.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import RobustScaler

from .config import BUSES, CLASS_NAMES, FEATURES, FIGURE_DIR, LABEL, N_SAMPLES, RF_GRID, STABLE, UNSTABLE

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
STABLE_COLOR = "#2a78d6"
UNSTABLE_COLOR = "#eb6834"
CLASS_COLORS = {STABLE: STABLE_COLOR, UNSTABLE: UNSTABLE_COLOR}
METRIC_COLOR = "#6250d6"
METRIC_COLOR_2 = "#1baf7a"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ORANGE_RAMP = LinearSegmentedColormap.from_list(
    "orange_ramp", ["#fdeae1", "#f8c4ad", "#f29772", "#eb6834", "#c4501f", "#8f3a14"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"])


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.markersize": 7,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.titlesize": 13, "figure.titleweight": "semibold",
    })


def save(fig, name, folder=None):
    folder = folder or FIGURE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


def _class_legend(ax_or_fig, **kw):
    handles = [Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in (STABLE, UNSTABLE)]
    return ax_or_fig.legend(handles=handles, **kw)


def _cell_text_color(value, vmin, vmax):
    return "white" if (value - vmin) / max(vmax - vmin, 1e-9) > 0.55 else INK


def _no_frame(ax):
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- data exploration
def plot_class_balance(df):
    order = [STABLE, UNSTABLE]
    counts = df[LABEL].value_counts()
    vals = [int(counts.get(c, 0)) for c in order]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(7.5, 2.3))
    ax.barh(range(2), vals, color=[CLASS_COLORS[c] for c in order], height=0.55,
            edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(2), [CLASS_NAMES[c] for c in order])
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + total * 0.01, i, f"{v:,}  ({v / total:.1%})", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("cases")
    ax.set_title(f"Class balance - {vals[1] / total:.0%} of cases are unstable")
    return fig


def plot_unstable_rate(df, by, order=None, title=None):
    """Horizontal bars: % of unstable cases per value of `by`."""
    g = df.groupby(by)[LABEL]
    rate = g.apply(lambda s: 100 * (s == UNSTABLE).mean())
    n = g.size()
    rate = rate.reindex(order) if order else rate.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(rate) + 1.2))
    ax.barh(range(len(rate)), rate.values, color=UNSTABLE_COLOR, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(rate)), [str(i).replace("_", " ") for i in rate.index])
    ax.invert_yaxis()
    for i, (k, r) in enumerate(rate.items()):
        ax.text(r + 1, i, f"{r:.1f}%", va="center", color=INK, fontsize=9)
        ax.text(r + 1 + 9, i, f"n = {n[k]:,}", va="center", color=MUTED, fontsize=8)
    ax.set_xlim(0, min(100, rate.max() * 1.25 + 18))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("unstable cases (%)")
    ax.set_title(title or f"Unstable cases by {by}")
    return fig


def plot_fault_time_heatmap(df):
    """Unstable % for each fault kind and fault duration."""
    d = df.dropna(subset=["Fault_Time_s"])
    rows = [k for k in ["SLG", "LLG", "3PHG", "Line trip"] if k in set(d["Fault_Kind"])]
    pv = d.pivot_table(index="Fault_Kind", columns="Fault_Time_s", values=LABEL,
                       aggfunc=lambda s: 100 * (s == UNSTABLE).mean()).reindex(rows)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    im = ax.imshow(pv.values, cmap=ORANGE_RAMP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(pv.shape[1]), [f"{t:g} s" for t in pv.columns])
    ax.set_yticks(range(pv.shape[0]), pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=10, color=_cell_text_color(v, 0, 100))
    _no_frame(ax)
    ax.set_xlabel("fault duration")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("unstable cases (%)", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Longer and more severe faults are more often unstable")
    return fig


def plot_rate_vs_load(df):
    """Small multiples: unstable % against load change, one panel per contingency type."""
    types = sorted(df["Contingency_Type"].unique())
    fig, axes = plt.subplots(1, len(types), figsize=(3.0 * len(types), 3.0), sharey=True)
    ymax = 0
    for ax, t in zip(np.atleast_1d(axes), types):
        d = df[df["Contingency_Type"] == t]
        r = d.groupby("Load_Change_pct")[LABEL].apply(lambda s: 100 * (s == UNSTABLE).mean()).sort_index()
        ymax = max(ymax, r.max())
        ax.plot(r.index, r.values, color=UNSTABLE_COLOR, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5)
        for x in (r.index[0], r.index[-1]):
            ax.annotate(f"{r[x]:.0f}%", (x, r[x]), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=INK_2)
        ax.set_title(t.replace("_", " "), fontsize=10)
        ax.set_xticks(r.index, [f"{x:+g}%" if x else "0%" for x in r.index])
        ax.set_xlabel("load change")
    np.atleast_1d(axes)[0].set_ylabel("unstable cases (%)")
    np.atleast_1d(axes)[0].set_ylim(0, ymax * 1.25 + 5)
    fig.suptitle("Higher load makes the system less stable", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_sample_traces(df, quantity="voltage"):
    """Median (line) and middle 50% (band) of each bus over the 4 samples, stable vs unstable."""
    tag, unit = ("Vmag", "voltage (p.u.)") if quantity == "voltage" else ("FreqHz", "frequency (Hz)")
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.4), sharex=True, sharey=True)
    x = np.arange(1, N_SAMPLES + 1)
    for ax, bus in zip(axes.ravel(), BUSES):
        cols = [f"{bus}_{tag}_{k}" for k in x]
        for c in (STABLE, UNSTABLE):
            v = df.loc[df[LABEL] == c, cols]
            ax.fill_between(x, v.quantile(0.25), v.quantile(0.75), color=CLASS_COLORS[c], alpha=0.18, linewidth=0)
            ax.plot(x, v.median(), color=CLASS_COLORS[c], marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_title(bus, fontsize=10)
        ax.set_xticks(x)
    for ax in axes[1]:
        ax.set_xlabel("sample")
    for ax in axes[:, 0]:
        ax.set_ylabel(unit)
    _class_legend(fig, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(f"Generator {quantity} over the 4 samples (line = median, band = middle 50% of cases)",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_extreme_values(df):
    """Largest voltage and largest frequency deviation of every case, to show the impossible readings."""
    vcols = [c for c in FEATURES if "_Vmag_" in c]
    fcols = [c for c in FEATURES if "_FreqHz_" in c]
    vmax = df[vcols].max(axis=1)
    fdev = (df[fcols] - 50).abs().max(axis=1)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, vals, limit, ylabel, lim_label in (
            (axes[0], vmax, 1.1, "largest voltage in the case (p.u.)", "1.1 p.u."),
            (axes[1], fdev, 5.0, "largest |f - 50 Hz| in the case (Hz)", "5 Hz")):
        for i, c in enumerate((STABLE, UNSTABLE)):
            m = df[LABEL] == c
            ax.scatter(i + rng.uniform(-0.28, 0.28, m.sum()), vals[m], s=14, color=CLASS_COLORS[c],
                       alpha=0.55, edgecolor=SURFACE, linewidth=0.3)
        ax.axhline(limit, color=INK_2, linewidth=0.9)
        ax.text(1.58, limit * 1.06, lim_label, ha="right", va="bottom", color=INK_2, fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks([0, 1], [CLASS_NAMES[STABLE], CLASS_NAMES[UNSTABLE]])
        ax.set_xlim(-0.5, 1.6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{int((vals > limit).sum())} cases above {lim_label}", fontsize=10)
    fig.suptitle("Impossible readings (kept in the data for this first try) - all in unstable cases",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_correlation(df):
    corr = df[FEATURES].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    centers = np.arange(N_SAMPLES / 2 - 0.5, len(FEATURES), N_SAMPLES)
    labels = [f"{b} V" for b in BUSES] + [f"{b} f" for b in BUSES]
    ax.set_xticks(centers, labels, rotation=90, fontsize=8)
    ax.set_yticks(centers, labels, fontsize=8)
    half = len(FEATURES) / 2 - 0.5
    ax.axhline(half, color=SURFACE, linewidth=2)
    ax.axvline(half, color=SURFACE, linewidth=2)
    _no_frame(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("correlation", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Correlation of the 80 inputs (each label covers its 4 samples)")
    return fig


def plot_pca(df):
    """2-D projection (PCA) for visualisation, coloured by class.

    Inputs are scaled by median / interquartile range and limited to +/-20 for this plot only,
    so the 10 impossible readings do not squeeze all other cases into one corner.
    """
    Z = np.clip(RobustScaler().fit_transform(df[FEATURES]), -20, 20)
    pca = PCA(n_components=2, random_state=0).fit(Z)
    P = pca.transform(Z)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for c in (STABLE, UNSTABLE):
        m = (df[LABEL] == c).to_numpy()
        ax.scatter(P[m, 0], P[m, 1], s=16, color=CLASS_COLORS[c], alpha=0.7, edgecolor=SURFACE,
                   linewidth=0.4, label=CLASS_NAMES[c])
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"principal component 1 ({ev[0]:.0%} of variance)")
    ax.set_ylabel(f"principal component 2 ({ev[1]:.0%} of variance)")
    ax.legend(loc="best")
    ax.set_title("Many unstable cases stand apart even in 2-D; the rest overlap the stable cluster")
    return fig


# ---------------------------------------------------------------- split and tuning
def plot_split_composition(summary):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    names = list(summary.index)
    for i, name in enumerate(names):
        s, u = summary.loc[name, "stable"], summary.loc[name, "unstable"]
        ax.barh(i, s, color=STABLE_COLOR, height=0.55, edgecolor=SURFACE, linewidth=2)
        ax.barh(i, u, left=s, color=UNSTABLE_COLOR, height=0.55, edgecolor=SURFACE, linewidth=2)
        r = summary.loc[name]
        ax.text(s + u + 30, i, f"{int(r['rows']):,} rows  |  {int(r['contingencies'])} contingencies  |  "
                f"{r['unstable_pct']:.1f}% unstable", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, summary["rows"].max() * 1.75)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("cases")
    _class_legend(ax, loc="lower right", ncol=2)
    ax.set_title("Train / validation / test split (by contingency, same unstable share)")
    return fig


def plot_rf_grid(results, best):
    """Cross-validated balanced accuracy: max_depth (rows) x min_samples_leaf (columns),
    one panel per max_features, for the chosen class_weight."""
    feats = [str(f) for f in RF_GRID["max_features"]]
    depths = [str(d) for d in RF_GRID["max_depth"]]
    leaves = list(RF_GRID["min_samples_leaf"])
    sub = results[results["class_weight"] == best["class_weight"]]
    vmin, vmax = sub["cv_bal_acc"].min(), sub["cv_bal_acc"].max()
    fig, axes = plt.subplots(1, len(feats), figsize=(4.3 * len(feats), 3.9), layout="constrained")
    for ax, mf in zip(np.atleast_1d(axes), feats):
        pv = sub[sub["max_features"] == mf].pivot_table(index="max_depth", columns="min_samples_leaf",
                                                         values="cv_bal_acc").reindex(index=depths, columns=leaves)
        im = ax.imshow(pv.values, cmap=BLUE_RAMP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(leaves)), [str(v) for v in leaves])
        ax.set_yticks(range(len(depths)), ["no limit" if d == "None" else d for d in depths])
        ax.set_xlabel("min_samples_leaf")
        ax.set_ylabel("max_depth")
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=_cell_text_color(v, vmin, vmax))
        if str(best["max_features"]) == mf:
            i, j = depths.index(str(best["max_depth"])), leaves.index(best["min_samples_leaf"])
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2.2))
        _no_frame(ax)
        ax.set_title(f"max_features = {mf}", fontsize=10)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("cross-validated balanced accuracy", color=INK_2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Random forest settings by cross-validation (class_weight = {best['class_weight']}, "
                 f"black box = chosen)", x=0.01, ha="left")
    return fig


def plot_learning_curve(lc):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for col, color, name in (("train_bal_acc", METRIC_COLOR, "Train"), ("val_bal_acc", METRIC_COLOR_2, "Validation")):
        ax.plot(lc["training_rows"], lc[col], color=color, marker="o", markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=name)
        ax.annotate(f"{name} {lc[col].iloc[-1]:.3f}", (lc["training_rows"].iloc[-1], lc[col].iloc[-1]),
                    textcoords="offset points", xytext=(8, 0), va="center", fontsize=9, color=INK)
    ax.set_xlim(0, lc["training_rows"].max() * 1.3)
    lo = min(lc["val_bal_acc"].min(), lc["train_bal_acc"].min())
    ax.set_ylim(max(0.5, lo - 0.03), 1.005)
    ax.set_xlabel("training rows used")
    ax.set_ylabel("balanced accuracy")
    ax.legend(loc="lower right")
    ax.set_title("Learning curve - does more training data still help?")
    return fig


# ---------------------------------------------------------------- results
def plot_metric_dots(table, metrics=("balanced_accuracy", "unstable_recall", "stable_recall", "unstable_precision")):
    """Dot plot of metrics (columns) for each set (rows of `table`)."""
    names = {"balanced_accuracy": "Balanced accuracy", "unstable_recall": "Unstable found (recall)",
             "stable_recall": "Stable found (recall)", "unstable_precision": "Unstable precision",
             "accuracy": "Accuracy", "roc_auc": "ROC-AUC"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 2.6), sharey=True)
    lo = min(0.9, float(table[list(metrics)].min().min()) - 0.02)
    for ax, m in zip(np.atleast_1d(axes), metrics):
        vals = table[m].to_numpy()
        y = np.arange(len(vals))
        ax.scatter(vals, y, s=70, color=METRIC_COLOR, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v, yi - 0.28, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xlim(lo, 1.0 + (1 - lo) * 0.08)
        ax.set_ylim(len(vals) - 0.5, -0.8)
        ax.set_yticks(y, list(table.index))
        ax.grid(axis="y", visible=False)
        ax.set_title(names.get(m, m), fontsize=10)
    fig.suptitle("Model results on each set", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_confusion_matrices(cms):
    """`cms`: dict title -> 2x2 DataFrame from evaluate.confusion (rows actual, cols predicted)."""
    fig, axes = plt.subplots(1, len(cms), figsize=(4.4 * len(cms), 3.9))
    for ax, (title, cm) in zip(np.atleast_1d(axes), cms.items()):
        counts = cm.to_numpy()
        pct = 100 * counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=BLUE_RAMP, vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{counts[i, j]:,}\n{pct[i, j]:.1f}%", ha="center", va="center",
                        fontsize=10, color=_cell_text_color(pct[i, j], 0, 100))
        ax.set_xticks(range(2), [c.replace("predicted ", "") for c in cm.columns])
        ax.set_yticks(range(2), [r.replace("actual ", "") for r in cm.index])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        _no_frame(ax)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Confusion matrices (% of each actual class)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_roc_pr(y_true, prob, name="Test set"):
    """ROC and precision-recall curves with Unstable as the positive class."""
    is_u = np.asarray(y_true) == UNSTABLE
    fpr, tpr, _ = roc_curve(is_u, prob)
    prec, rec, _ = precision_recall_curve(is_u, prob)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    a1.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    a1.plot(fpr, tpr, color=METRIC_COLOR)
    a1.set_xlabel("stable cases wrongly flagged (false positive rate)")
    a1.set_ylabel("unstable cases found (true positive rate)")
    a1.set_title(f"ROC curve - AUC {roc_auc_score(is_u, prob):.4f}", fontsize=10)
    a1.set_aspect("equal")
    base = is_u.mean()
    a2.axhline(base, color=AXIS, linewidth=1)
    a2.text(0.02, base + 0.02, f"no-skill level {base:.2f}", fontsize=8, color=MUTED)
    a2.plot(rec, prec, color=METRIC_COLOR)
    a2.set_xlabel("unstable cases found (recall)")
    a2.set_ylabel("flagged cases truly unstable (precision)")
    a2.set_ylim(0, 1.03)
    a2.set_title(f"Precision-recall - average precision {average_precision_score(is_u, prob):.4f}", fontsize=10)
    fig.suptitle(f"{name}: how well the forest's probability ranks unstable cases", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_probability_distribution(y_true, prob, name="Test set"):
    y = np.asarray(y_true)
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for c in (STABLE, UNSTABLE):
        ax.hist(prob[y == c], bins=bins, color=CLASS_COLORS[c], alpha=0.55, edgecolor=SURFACE,
                linewidth=0.6, label=CLASS_NAMES[c])
    ax.axvline(0.5, color=INK, linewidth=1)
    ax.text(0.5, ax.get_ylim()[1] * 0.97, "  decision threshold 0.5", va="top", fontsize=8, color=INK_2)
    ax.set_xlabel("predicted probability of instability  (> 0.5 predicted Unstable)")
    ax.set_ylabel("cases")
    ax.legend(loc="upper center")
    ax.set_title(f"{name}: random forest probability by actual class")
    return fig


def plot_breakdown_dots(table, title):
    """Accuracy per group (dot), with case counts and errors written beside it."""
    t = table.copy()
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(t) + 1.4))
    y = np.arange(len(t))
    ax.scatter(t["accuracy"], y, s=70, color=METRIC_COLOR, edgecolor=SURFACE, linewidth=1.5, zorder=3, clip_on=False)
    lo = min(0.85, t["accuracy"].min() - 0.03)
    for yi, (_, r) in zip(y, t.iterrows()):
        # x in axes fraction (right of the plot), y in data units
        ax.text(1.04, yi, f"{r['accuracy']:.3f}   n = {int(r['cases'])}, missed unstable {int(r['unstable_missed'])}, "
                f"false alarms {int(r['false_alarms'])}", va="center", fontsize=8, color=INK_2,
                transform=ax.get_yaxis_transform())
    ax.set_xlim(lo, 1.0)
    ax.set_yticks(y, [str(i).replace("_", " ") for i in t.index])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy")
    ax.set_title(title)
    return fig


def plot_importance(imp, top=20, xlabel="drop in test balanced accuracy when this measurement is shuffled",
                    title="Which generator measurements the forest relies on"):
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.3))
    ax.barh(range(len(t)), t["importance"], xerr=t["std"], color=METRIC_COLOR, height=0.65,
            edgecolor=SURFACE, linewidth=1.5, error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 0})
    ax.set_yticks(range(len(t)), t["feature_group"])
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return fig
