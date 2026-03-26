import numpy as np


def apply_threshold(scores, threshold, higher_is_same=True):
    scores = np.asarray(scores)

    if higher_is_same:
        predictions = (scores >= threshold).astype(int)
    else:
        predictions = (scores <= threshold).astype(int)

    return predictions


def compute_confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    return {
        "tp": int(true_positive),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative)
    }


def compute_accuracy(confusion):
    total = confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"]

    if total == 0:
        raise ValueError("Confusion matrix total cannot be zero")

    return (confusion["tp"] + confusion["tn"]) / total


def compute_precision(confusion):
    denominator = confusion["tp"] + confusion["fp"]

    if denominator == 0:
        return 0.0

    return confusion["tp"] / denominator


def compute_recall(confusion):
    denominator = confusion["tp"] + confusion["fn"]

    if denominator == 0:
        return 0.0

    return confusion["tp"] / denominator


def compute_f1_score(confusion):
    precision = compute_precision(confusion)
    recall = compute_recall(confusion)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_balanced_accuracy(confusion):
    positive_total = confusion["tp"] + confusion["fn"]
    negative_total = confusion["tn"] + confusion["fp"]

    if positive_total == 0 or negative_total == 0:
        return 0.0

    true_positive_rate = confusion["tp"] / positive_total
    true_negative_rate = confusion["tn"] / negative_total

    return (true_positive_rate + true_negative_rate) / 2.0
