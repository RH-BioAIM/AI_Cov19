"""
Runs the nested cross-validation hyperparameter search shared by the
restricted clinical-only and integrated models.
"""
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error

GRID = [
    {"max_depth": d, "eta": e, "min_child_weight": m}
    for d, e, m in itertools.product([3, 4, 6], [0.03, 0.1, 0.3], [1, 5])
]

BASE_PARAMS = {"objective": "reg:squarederror", "tree_method": "hist", "seed": 42, "nthread": 4}
# nthread pinned low deliberately: on this shared node, xgboost's default of
# spawning threads = all visible cores (64) causes severe contention on data
# this small (rows in the hundreds) -- measured 16.7s for 5 boosting rounds
# at nthread=64 vs 0.09s at nthread=4 on identical data/params (~180x).
# Nested search runs hundreds of small fits; nthread=4 is what makes it
# tractable at all.
SEARCH_NUM_BOOST_ROUND = 3000
SEARCH_EARLY_STOPPING_ROUNDS = 50
FINAL_REFIT_MIN_ROUNDS = 5  # floor, in case a fast-eta combo's mean best_iteration rounds to ~0


def _train_one(params, dtrain, dtest, num_boost_round, early_stopping_rounds):
    full_params = {**BASE_PARAMS, **params}
    evals = [(dtest, "eval")]
    kwargs = dict(num_boost_round=num_boost_round, evals=evals, verbose_eval=False)
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    model = xgb.train(full_params, dtrain, **kwargs)
    best_iter = model.best_iteration if early_stopping_rounds is not None else num_boost_round - 1
    return model, best_iter


def inner_cv_score(X_train_outer, y_train_outer, inner_folds, combo):
    """4-fold inner CV within one outer fold's training set. Returns
    (mean_c_index, mean_rmse, mean_best_iteration_plus_1)."""
    c_indices, rmses, iters = [], [], []
    for g in sorted(inner_folds.unique()):
        tr_idx = inner_folds != g
        va_idx = inner_folds == g
        dtr = xgb.DMatrix(X_train_outer[tr_idx], label=y_train_outer[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(X_train_outer[va_idx], label=y_train_outer[va_idx], enable_categorical=True)
        model, best_iter = _train_one(combo, dtr, dva, SEARCH_NUM_BOOST_ROUND, SEARCH_EARLY_STOPPING_ROUNDS)
        pred = model.predict(dva, iteration_range=(0, best_iter + 1))
        y_va = y_train_outer[va_idx]
        c_indices.append(concordance_index(y_va, pred))
        rmses.append(float(np.sqrt(mean_squared_error(y_va, pred))))
        iters.append(best_iter + 1)
    return float(np.mean(c_indices)), float(np.mean(rmses)), float(np.mean(iters))


def nested_cv_select_and_fit(X, y, outer_folds, label):
    """Full nested CV for one model. Returns (oof_pred: pd.Series,
    per_outer_fold_log: list[dict], selection_table: pd.DataFrame with every
    combo's inner score for every outer fold)."""
    oof_pred = pd.Series(index=X.index, dtype=float)
    fold_log = []
    selection_rows = []

    for f in sorted(outer_folds.unique()):
        train_idx = outer_folds != f
        test_idx = outer_folds == f
        X_tr_outer, y_tr_outer = X[train_idx], y[train_idx]
        inner_folds = outer_folds[train_idx]  # the other 4 fold labels, used as inner CV splits

        best_combo, best_cidx, best_rmse, best_rounds = None, -np.inf, None, None
        for combo in GRID:
            mean_c, mean_rmse, mean_rounds = inner_cv_score(X_tr_outer, y_tr_outer, inner_folds, combo)
            selection_rows.append({
                "model": label, "outer_fold": f, **combo,
                "inner_mean_c_index": mean_c, "inner_mean_rmse": mean_rmse, "inner_mean_rounds": mean_rounds,
            })
            if mean_c > best_cidx:
                best_combo, best_cidx, best_rmse, best_rounds = combo, mean_c, mean_rmse, mean_rounds

        final_rounds = max(FINAL_REFIT_MIN_ROUNDS, int(round(best_rounds)))
        dtrain = xgb.DMatrix(X_tr_outer, label=y_tr_outer, enable_categorical=True)
        dtest = xgb.DMatrix(X[test_idx], label=y[test_idx], enable_categorical=True)
        final_model, _ = _train_one(best_combo, dtrain, dtest, final_rounds, early_stopping_rounds=None)
        oof_pred[test_idx] = final_model.predict(dtest, iteration_range=(0, final_rounds))

        fold_log.append({
            "model": label, "outer_fold": f, "n_train": int(train_idx.sum()), "n_test": int(test_idx.sum()),
            **best_combo, "selected_by_inner_c_index": best_cidx, "selected_by_inner_rmse": best_rmse,
            "final_num_boost_round": final_rounds,
        })
        print(f"  [{label}] outer fold {f}: selected {best_combo} "
              f"(inner mean C-index={best_cidx:.4f}, inner mean RMSE={best_rmse:.4f}) "
              f"-> final refit with num_boost_round={final_rounds}")

    assert oof_pred.notna().all(), "every row must get exactly one OOF prediction"
    return oof_pred, pd.DataFrame(fold_log), pd.DataFrame(selection_rows)


def modal_combo(fold_log_df):
    """Most frequently selected combo across outer folds -- the single
    config reported as 'the model's hyperparameters'."""
    key_cols = ["max_depth", "eta", "min_child_weight"]
    counts = fold_log_df.groupby(key_cols).size().reset_index(name="n_folds_selected")
    counts = counts.sort_values("n_folds_selected", ascending=False)
    top = counts.iloc[0]
    return {c: top[c] for c in key_cols}, counts
