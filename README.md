# Transient stability prediction - IEEE 39-bus system (Random forest)

Predicts whether the IEEE 39-bus system stays **stable** after a contingency from 4 voltage and 4 frequency samples
at each of the 10 generator buses (B30-B39), i.e. 80 inputs. Label `Overall_Stability` comes from the TSI
(1 = stable, 0 = unstable).

**First try:** all 3,715 cases, outliers kept (the contingencies with impossible readings are in the training set).

## Contents

| Path | What it is |
|---|---|
| `random_forest_stability.ipynb` | Full analysis with all graphs - start here |
| `rf_stability/` | Python package used by the notebook and scripts (`config`, `data`, `model`, `evaluate`, `plots`, `predict`) |
| `train_random_forest.py` | Runs the whole procedure from the command line and saves model, figures and reports |
| `predict_rf.py` | Predicts new cases from a CSV |
| `IEEE39_voltage_frequency_4sample_dataset.csv` | The dataset |
| `models/stability_rf.joblib` | Trained random forest |
| `figures/` | All graphs (PNG) |
| `reports/` | Text report, tuning grid, test-set predictions |
| `examples/` | Example input files for `predict_rf.py` |
| `paper/` | IEEE conference paper (max. 6 pages) with its figures and the compiled PDF |

## Procedure

1. Split by `Contingency_ID` into train / validation / test (about 70 / 15 / 15) - all 5 load levels of a
   contingency stay in one set.
2. Choose `max_depth`, `min_samples_leaf`, `max_features` and `class_weight` of a 300-tree random forest by grouped
   5-fold cross-validation inside the training set.
3. Check the chosen forest on the validation set.
4. Refit on train + validation, evaluate once on the test set.

## Run (from this folder)

```
py -m pip install -r requirements.txt
py train_random_forest.py
py predict_rf.py examples\example_case_unstable.csv
```

`predict_rf.py` accepts either one row per case with the 80 training column names, or one case as a 10-row table:

```
Bus,V1,V2,V3,V4,F1,F2,F3,F4
B30,...
...
B39,...
```

It reports the prediction, the predicted probability of instability (`P_unstable`, > 0.5 means unstable) and any
inputs outside the range seen in training.

The saved model must be loaded with the scikit-learn version it was trained with (1.9.1, see `requirements.txt`).
