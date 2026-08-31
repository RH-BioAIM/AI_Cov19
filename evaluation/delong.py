"""
Implements the fast DeLong algorithm for comparing correlated ROC AUCs.
Provides delong_paired_test and delong_auc_ci.
"""
import numpy as np


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1]), "ground_truth must be binary 0/1"
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_paired_test(ground_truth, predictions_one, predictions_two):
    """Paired DeLong test for two correlated ROC curves on the SAME patients.

    Returns dict with auc_one, auc_two, auc_diff, ci95_diff (tuple), z, p_value.
    """
    ground_truth = np.asarray(ground_truth)
    predictions_one = np.asarray(predictions_one, dtype=float)
    predictions_two = np.asarray(predictions_two, dtype=float)
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    preds = np.vstack((predictions_one, predictions_two))[:, order]

    aucs, delongcov = fastDeLong(preds, label_1_count)
    auc1, auc2 = aucs[0], aucs[1]
    var1, var2, cov12 = delongcov[0, 0], delongcov[1, 1], delongcov[0, 1]

    diff = auc1 - auc2
    var_diff = var1 + var2 - 2 * cov12
    se_diff = np.sqrt(max(var_diff, 0.0))

    z = diff / se_diff if se_diff > 0 else 0.0
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    ci_low, ci_high = diff - 1.96 * se_diff, diff + 1.96 * se_diff

    return {
        "auc_one": auc1, "auc_two": auc2, "auc_diff": diff,
        "se_diff": se_diff, "ci95_low": ci_low, "ci95_high": ci_high,
        "z": z, "p_value": p_value,
    }


def delong_auc_ci(ground_truth, predictions):
    """Single-model AUC with DeLong (1988) variance -> 95% CI."""
    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions, dtype=float)
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    preds = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(preds, label_1_count)
    auc = aucs[0]
    se = np.sqrt(max(delongcov[0, 0], 0.0))
    return {"auc": auc, "se": se, "ci95_low": auc - 1.96 * se, "ci95_high": auc + 1.96 * se}
