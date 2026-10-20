"""Build code/signal3_refit.ipynb -- Lender 3 refit: train ONLY CatBoost on the full data.

This replaces the verbatim copy of signal3_models (which still ran the full logreg / spline /
xgboost / catboost comparison and the CV-only refit). The refit notebook keeps exactly one model
family (CatBoost), scores it once on the test split, then refits on ALL past loans (train + test
= 100k) and predicts PD for the new applications. Export keeps the agreed format id,pd,pred,threshold.
"""
from pathlib import Path

import nbformat as nbf

SEED = 42
TAG = "catboost"

md_intro = """# Signal 3 — refit on the full data

We are **Lender 3**: for new applicants we see the shared features (sex, employment, married, income) plus our own `signal3`. This notebook is the *deployment* stage of the Lender 3 pipeline: the model comparison in `signal3_models` selected **CatBoost**, so here we

1. re-tune CatBoost on the shared train split by 5-fold CV log loss (train only — the test split is never used for tuning),
2. score it **once** on the test split as the performance estimate,
3. **refit the same, already-tuned CatBoost on all 100,000 past loans** (train + test — no re-tuning) and
4. predict the default probability (PD) of the new applications.

- `sex` is **not** used as a feature.
- `signal1` / `signal2` are not used: new applicants only have `signal3`."""

md_load = """## 1. Load the shared split and the new applications

`train.csv` / `test.csv` are the shared 70k/30k split produced in `eda_and_split`; the new applications are the 100k applicants we price this round. A borrower never appears in both splits."""

md_features = """## 2. Features

Transforms below work row by row and learn nothing from the data, so they are applied once, identically, to train, test and the new applications:

- `employment` is used by CatBoost as a **categorical feature directly** (no one-hot): a single split can separate any one employment group.
- `log_income = log(1 + income)` — income is right-skewed and top-coded at 1M € — plus a `zero_income` flag (~17% of loans, with a much higher default rate).
- a `signal3_zero` flag, since ~20% of `signal3` is exactly 0 (a mass point).

Transforms that learn from the data (income scaling) live inside the model's pipeline and are fit on the training folds only."""

md_tune = """## 3. Tune CatBoost on CV log loss

Optuna tunes the hyperparameters on the same 5-fold CV log loss as `signal3_models`; early stopping sets the number of trees. The test split is scored only after tuning."""

md_test = """## 4. Score once on the test split

CatBoost is fit on the train split only and scored once on the test split (out-of-sample log loss / Brier / AUC). This is the performance estimate."""

md_refit = """## 5. Refit on the full data and predict the new applications

The tuned CatBoost is refit — **with the same settings, no re-tuning** — on all 100,000 past loans (train + test), then predicts the default probability of the new applications orthodoxly.

Because the new applicants come from the same population, the mean predicted PD should be close to the past default rate (~10%)."""

md_export = """## 6. Export

Prediction file uses the agreed format `id, pd, pred, threshold`; the 0.5 threshold is for the format only — pricing uses `pd`."""

code_imports = """import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv as catboost_cv
from catboost import __version__ as catboost_version
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")

TARGET = "default"
SEED = 42
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

EMPLOYMENT_LEVELS = ["employed", "retired", "student", "unemployed"]
EMPLOYMENT_ALL = [f"employment_{level}" for level in EMPLOYMENT_LEVELS]

CATBOOST_FEATURES = ["employment", "married", "income", "signal3"]
CATBOOST_CATEGORICAL = ["employment"]

CB_BASE = {"loss_function": "Logloss", "bootstrap_type": "Bernoulli", "random_seed": SEED,
           "logging_level": "Silent", "allow_writing_files": False}
N_TRIALS = 40
SEED2 = 42"""

code_prepare = """def prepare(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Model features of a loans or applications table (row-wise, nothing fitted).\"\"\"
    unknown = set(df["employment"]) - set(EMPLOYMENT_LEVELS)
    assert not unknown, f"unknown employment levels: {unknown}"
    X = pd.DataFrame(index=df.index)
    X["employment"] = df["employment"]
    X["married"] = df["married"]
    X["income"] = df["income"]
    X["log_income"] = np.log1p(df["income"])
    X["zero_income"] = (df["income"] == 0).astype(int)
    X["signal3"] = df["signal3"]
    X["signal3_zero"] = (df["signal3"] == 0).astype(int)
    return X


# CatBoost uses employment as a category directly
CB_FEATURES = ["employment", "married", "income", "signal3"]
CB_CATEGORICAL = ["employment"]"""

code_load = """train = pd.read_csv(PROCESSED_DIR / "train.csv")
test = pd.read_csv(PROCESSED_DIR / "test.csv")
new = pd.read_csv(RAW_DIR / "NewApplications_Lender3_Round1.csv")

for name, part in [("train", train), ("test", test)]:
    print(f"{name}: {len(part):,} rows, default rate {part[TARGET].mean():.2%}")
print(f"new applications: {len(new):,} rows")

# Check the files come from the shared split
assert (len(train), len(test)) == (70_000, 30_000), "unexpected split sizes: re-run eda_and_split.ipynb"
assert not set(train["id"]) & set(test["id"]), "a borrower appears in both train and test"
assert new["id"].is_unique and not new.isna().any().any(), "new applications: duplicate ids or missing values"

X_train, y_train = prepare(train), train[TARGET]
X_test, y_test = prepare(test), test[TARGET]
X_new = prepare(new)
fold_list = list(folds.split(X_train, y_train))
print(f"train features: {list(X_train.columns)}")"""

code_tune = """train_pool = Pool(X_train[CB_FEATURES], label=y_train, cat_features=CB_CATEGORICAL)


def catboost_objective(trial: optuna.Trial) -> float:
    params = CB_BASE | {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 2, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    cv = catboost_cv(Pool(X_train[CB_FEATURES], y_train, cat_features=CB_CATEGORICAL), params,
                     fold_count=5, partition_random_seed=SEED, early_stopping_rounds=50)
    best_iter = cv["test-Logloss-mean"].idxmin()
    trial.set_user_attr("n_rounds", int(cv.loc[best_iter, "iterations"]) + 1)
    trial.set_user_attr("cv_std", cv.loc[best_iter, "test-Logloss-std"])
    return cv.loc[best_iter, "test-Logloss-mean"]


optuna.logging.set_verbosity(optuna.logging.WARNING)
cb_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
cb_study.optimize(catboost_objective, n_trials=N_TRIALS)

cb_params = CB_BASE | cb_study.best_params
cb_rounds = cb_study.best_trial.user_attrs["n_rounds"]
print(f"Best CV log loss {cb_study.best_value:.4f} with {cb_rounds} trees: {cb_study.best_params}")"""

code_test = """model = CatBoostClassifier(**cb_params, iterations=cb_rounds)
model.fit(X_train[CB_FEATURES], y_train, cat_features=CB_CATEGORICAL)
p_test = model.predict_proba(X_test[CB_FEATURES])[:, 1]

print(f"test log loss {log_loss(y_test, p_test):.4f} | brier {brier_score_loss(y_test, p_test):.4f} "
      f"| auc {roc_auc_score(y_test, p_test):.4f} | mean PD {p_test.mean():.2%} vs observed {y_test.mean():.2%}")"""

code_refit = """# Refit on all past loans, then predict the new applications
past = pd.concat([train, test], ignore_index=True)
model_full = CatBoostClassifier(**cb_params, iterations=cb_rounds)
model_full.fit(prepare(past)[CB_FEATURES], past[TARGET], cat_features=CB_CATEGORICAL, verbose=False)
p_new = model_full.predict_proba(X_new[CB_FEATURES])[:, 1]

new_predictions = pd.DataFrame({
    "id": new["id"].to_numpy(),
    "pd": p_new,
    "pred": (p_new >= 0.5).astype(int),
    "threshold": 0.5,
})
assert len(new_predictions) == len(new) and new_predictions["pd"].between(0, 1).all()
new_path = PROCESSED_DIR / f"signal3_new_applications_predictions_{TAG}.csv"
new_predictions.to_csv(new_path, index=False)
print(f"{new_path.name}: {len(new_predictions):,} rows | mean PD {new_predictions['pd'].mean():.2%} "
      f"vs past default rate {past[TARGET].mean():.2%}")
new_predictions["pd"].describe().round(4)"""

cells = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(md_load),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell(md_features),
    nbf.v4.new_code_cell(code_prepare),
    nbf.v4.new_markdown_cell(md_tune),
    nbf.v4.new_code_cell(code_tune),
    nbf.v4.new_markdown_cell(md_test),
    nbf.v4.new_code_cell(code_test),
    nbf.v4.new_markdown_cell(md_refit),
    nbf.v4.new_code_cell(code_refit),
    nbf.v4.new_markdown_cell(md_export),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.10"}}
out = Path("code/signal3_refit.ipynb")
out.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {out} with {len(cells)} cells")
