"""Overall and fairness metrics for binary glaucoma detection (ported from FairVision).

All functions take predicted probabilities of shape [N, 2] (softmax output).
Uni-modal baselines output one sigmoid probability per sample; `two_column` converts
those to [1 - p, p] first.
"""
import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import auc, roc_curve


def two_column(probs):
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return np.stack([1.0 - probs, probs], axis=1)
    return probs


def accuracy(probs, gts):
    return float(np.mean(np.argmax(probs, axis=1) == gts))


def auc_score(probs, gts):
    fpr, tpr, _ = roc_curve(np.ravel(gts), probs[:, 1])
    return auc(fpr, tpr)


def equity_scaled_accuracy(probs, gts, attr, alpha=1.0):
    overall = accuracy(probs, gts)
    gap = sum(abs(accuracy(probs[attr == g], gts[attr == g]) - overall) for g in np.unique(attr).astype(int))
    return overall / (alpha * gap + 1)


def equity_scaled_auc(probs, gts, attr, alpha=1.0):
    overall = auc_score(probs, gts)
    gap = sum(abs(auc_score(probs[attr == g], gts[attr == g]) - overall) for g in np.unique(attr).astype(int))
    return overall / (alpha * gap + 1)


def between_group_disparity(group_aucs, overall_auc):
    return np.std(group_aucs) / overall_auc, (np.max(group_aucs) - np.min(group_aucs)) / overall_auc


def evaluate_comprehensive_perf(probs, gts, attrs):
    """Per-attribute equity-scaled acc/AUC, group AUCs, DPD, EOD and AUC disparity.

    attrs: [num_attributes, N] integer array.
    """
    es_accs, es_aucs, aucs_by_attr, dpds, eods, disparities = [], [], [], [], [], []
    overall_auc = auc_score(probs, gts)
    pred_labels = np.argmax(probs, axis=1)

    for attr in attrs:
        es_accs.append(equity_scaled_accuracy(probs, gts, attr))
        es_aucs.append(equity_scaled_auc(probs, gts, attr))

        group_aucs = []
        for g in np.unique(attr).astype(int):
            try:
                group_aucs.append(auc_score(probs[attr == g], gts[attr == g]))
            except Exception:
                group_aucs.append(-1.0)
        aucs_by_attr.append(group_aucs)
        disparities.append(list(between_group_disparity(group_aucs, overall_auc)))

        dpds.append(demographic_parity_difference(gts, pred_labels, sensitive_features=attr))
        eods.append(equalized_odds_difference(gts, pred_labels, sensitive_features=attr))

    return es_accs, es_aucs, aucs_by_attr, dpds, eods, disparities
