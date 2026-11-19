"""Train, evaluate and save the transient-stability random forest (command-line version of the notebook).

Steps
  1. Load all 3,715 cases (first try: nothing removed, impossible readings kept).
  2. Split by contingency into train / validation / test (about 70 / 15 / 15); the contingencies
     with impossible readings are placed in the training set.
  3. Choose max_depth, min_samples_leaf, max_features and class_weight by grouped 5-fold
     cross-validation inside the training set.
  4. Check the chosen forest on the validation set.
  5. Refit on train + validation and evaluate once on the test set.
  6. Save the model, a text report, test predictions, example inputs and all figures.

Usage (from any folder)
    py train_random_forest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rf_stability import data, evaluate, model, plots  # noqa: E402
from rf_stability.config import (BUSES, DATA_FILE, EXAMPLE_DIR, GROUP, LABEL, MODEL_FILE,  # noqa: E402
                                 N_SAMPLES, REPORT_DIR, STABLE, UNSTABLE)


def write_example(row, path):
    """Save one case in the per-generator layout (10 rows: Bus,V1..V4,F1..F4)."""
    table = pd.DataFrame({"Bus": BUSES})
    for k in range(1, N_SAMPLES + 1):
        table[f"V{k}"] = [row[f"{b}_Vmag_{k}"] for b in BUSES]
    for k in range(1, N_SAMPLES + 1):
        table[f"F{k}"] = [row[f"{b}_FreqHz_{k}"] for b in BUSES]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def main():
    plots.apply_style()
    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    raw = data.load_dataset()
    log(f"Dataset: {DATA_FILE.name}")
    log(data.quality_report(raw).to_string())
    df = data.add_scenario_details(raw)

    train, val, test = data.split_train_val_test(df)
    summary = data.split_summary(train, val, test)
    log("\n== Split (by contingency) ==")
    log(summary.to_string())
    log("Rows with impossible readings (train / validation / test): "
        f"{[int(data.extreme_rows(s).sum()) for s in (train, val, test)]}")

    X_tr, y_tr = data.get_xy(train)
    log("\n== Choosing settings: grouped 5-fold cross-validation inside the training set ==")
    cv_results, best = model.tune_rf_with_cv(X_tr, y_tr, train[GROUP])
    log(cv_results.head(10).round(4).to_string(index=False))
    log(f"Chosen: {best}")

    selection_model = model.build_rf(**best, n_jobs=-1).fit(X_tr, y_tr)
    _, _, m_train = evaluate.evaluate(selection_model, train)
    val_pred, val_prob, m_val = evaluate.evaluate(selection_model, val)

    trainval = pd.concat([train, val])
    final = model.build_rf(**best, n_jobs=-1).fit(*data.get_xy(trainval))
    test_pred, test_prob, m_test = evaluate.evaluate(final, test)

    m_cv = {"balanced_accuracy": float(cv_results.iloc[0]["cv_bal_acc"]), "std": float(cv_results.iloc[0]["cv_std"])}
    table = pd.DataFrame({"Train": m_train, "Validation": m_val, "Test": m_test}).T
    log("\n== Results ==")
    log(f"Cross-validation on train (chosen setting): balanced accuracy {m_cv['balanced_accuracy']:.4f} "
        f"+/- {m_cv['std']:.4f}")
    log("Train and Validation rows: forest fitted on the training set.")
    log("Test row: final forest refitted on train + validation, scored once on the untouched test set.")
    log(table.round(4).to_string())
    log("\nValidation confusion matrix:")
    log(evaluate.confusion(val[LABEL], val_pred).to_string())
    log("\nTest confusion matrix:")
    log(evaluate.confusion(test[LABEL], test_pred).to_string())
    by_type = evaluate.breakdown(test, test_pred, "Contingency_Type")
    log("\nTest results by contingency type:")
    log(by_type.to_string())
    log("\nTest results by load change (%):")
    log(evaluate.breakdown(test, test_pred, "Load_Change_pct").to_string())
    for name, d, p, s in (("validation", val, val_pred, val_prob), ("test", test, test_pred, test_prob)):
        log(f"\n{name.capitalize()} errors grouped by contingency:")
        log(evaluate.errors_by_contingency(d, p, s).to_string())

    perm = evaluate.grouped_permutation_importance(final, test)
    impurity = evaluate.impurity_importance_by_group(final)
    lc = evaluate.learning_curve_on_validation(train, val, best)
    log("\nPermutation importance (top 10):")
    log(perm.head(10).round(4).to_string(index=False))
    log("\nImpurity importance (top 10):")
    log(impurity.head(10).round(4).to_string(index=False))
    log("\nLearning curve:")
    log(lc.round(4).to_string(index=False))

    model.save_model(final, best, {"cv": m_cv, "train": m_train, "validation": m_val, "test": m_test}, trainval,
                     notes="First try: all rows kept, outliers not removed. Fitted on train + validation.")
    log(f"\nModel saved   -> {MODEL_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = test[["Case_ID", "Operating_Condition", "Contingency_Type", "Contingency_ID", LABEL]].copy()
    out["Predicted"] = test_pred
    out["P_unstable"] = test_prob.round(4)
    out.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    cv_results.to_csv(REPORT_DIR / "cv_grid.csv", index=False)

    # Example inputs for predict_rf.py. They demonstrate the input format, so they are taken from
    # test cases (unseen by the final forest) that the forest classifies correctly.
    for cls, name in ((UNSTABLE, "unstable"), (STABLE, "stable")):
        row = test[(test[LABEL].to_numpy() == cls) & (test_pred == cls)].iloc[0]
        write_example(row, EXAMPLE_DIR / f"example_case_{name}.csv")
        log(f"Example input -> {EXAMPLE_DIR / f'example_case_{name}.csv'}  (actual: {name}, {row['Case_ID']})")

    figures = {
        "01_class_balance": plots.plot_class_balance(df),
        "02_unstable_by_contingency_type": plots.plot_unstable_rate(
            df, "Contingency_Type", title="Unstable cases by contingency type"),
        "03_unstable_by_fault_and_time": plots.plot_fault_time_heatmap(df),
        "04_unstable_vs_load": plots.plot_rate_vs_load(df),
        "05_voltage_traces": plots.plot_sample_traces(df, "voltage"),
        "06_frequency_traces": plots.plot_sample_traces(df, "frequency"),
        "07_impossible_readings": plots.plot_extreme_values(df),
        "08_correlation": plots.plot_correlation(df),
        "09_pca": plots.plot_pca(df),
        "10_split": plots.plot_split_composition(summary),
        "11_cv_grid": plots.plot_rf_grid(cv_results, best),
        "12_learning_curve": plots.plot_learning_curve(lc),
        "13_metrics_by_set": plots.plot_metric_dots(table),
        "14_confusion_matrices": plots.plot_confusion_matrices({
            "Validation (forest fitted on train)": evaluate.confusion(val[LABEL], val_pred),
            "Test (final forest)": evaluate.confusion(test[LABEL], test_pred)}),
        "15_roc_pr_test": plots.plot_roc_pr(test[LABEL], test_prob),
        "16_probability_distribution_test": plots.plot_probability_distribution(test[LABEL], test_prob),
        "17_test_by_contingency_type": plots.plot_breakdown_dots(by_type, "Test accuracy by contingency type"),
        "18_permutation_importance": plots.plot_importance(perm),
        "19_impurity_importance": plots.plot_importance(
            impurity, xlabel="mean decrease in Gini impurity (sum of the 4 samples)",
            title="Built-in (impurity) importance of the forest"),
    }
    for name, fig in figures.items():
        plots.save(fig, name)
        plt.close(fig)
    log(f"Figures saved -> {plots.FIGURE_DIR}")
    (REPORT_DIR / "training_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved  -> {REPORT_DIR / 'training_report.txt'}")


if __name__ == "__main__":
    main()
