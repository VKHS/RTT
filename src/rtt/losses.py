"""Task losses, probability floors and metrics without label-shape shortcuts."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def log_probabilities(logits, multilabel=False):
    """[items,classes] or [items,tasks,2], with binary outcomes ordered 0,1."""
    value = logits.double()
    if multilabel:
        return torch.stack((F.logsigmoid(-value), F.logsigmoid(value)), dim=-1)
    return F.log_softmax(value, dim=-1)


def unit_weights(mask, device=None):
    weights = mask.to(device=device, dtype=torch.float64)
    if not bool(weights.any()):
        raise ValueError("A scored call needs at least one observed task unit")
    return weights / weights.sum()


def supervised_loss(logits, labels, multilabel=False, mask=None):
    if labels is None:
        raise ValueError("Supervised loss is offline only and requires labels")
    if multilabel:
        observed = torch.isfinite(labels) if mask is None else mask
        losses = F.binary_cross_entropy_with_logits(logits.double(), torch.nan_to_num(labels.double()), reduction="none")
        return (losses * unit_weights(observed, logits.device)).sum()
    losses = F.cross_entropy(logits.double(), labels.long(), reduction="none")
    return losses.mean() if mask is None else (losses * unit_weights(mask, logits.device)).sum()


def floor_damage(reference_logits, logits, mask, multilabel=False):
    reference = log_probabilities(reference_logits, multilabel)
    candidate = log_probabilities(logits, multilabel)
    per_unit = (reference - candidate).amax(dim=-1)
    weights = unit_weights(mask, logits.device)
    return (per_unit * weights).sum(), per_unit, reference, candidate


def _average_precision(y, score):
    # Average precision with ties grouped exactly as threshold-based PR curves.
    order = np.argsort(-score, kind="stable")
    target, sorted_score = y[order], score[order]
    stops = np.r_[np.flatnonzero(np.diff(sorted_score)), len(score) - 1]
    true_positives = np.cumsum(target)[stops]
    increments = np.diff(np.r_[0, true_positives])
    return float(np.sum(increments * true_positives / (stops + 1)) / np.sum(target))


def task_metrics(labels, logits, *, multilabel=False):
    labels = np.asarray(labels)
    scores = np.asarray(logits)
    if multilabel:
        values = []
        for task in range(labels.shape[1]):
            known = np.isfinite(labels[:, task])
            target, score = labels[known, task], scores[known, task]
            if len(target) and np.any(target == 0) and np.any(target == 1):
                values.append(_average_precision(target, score))
        return {"macro_ap": float(np.mean(values)) if values else None,
                "ap_tasks_scored": len(values), "observed_task_units": int(np.isfinite(labels).sum())}
    prediction = scores.argmax(axis=1)
    output = {"accuracy": float(np.mean(prediction == labels)), "items": int(len(labels))}
    if scores.shape[1] == 2:
        positive = labels == 1
        chosen = prediction == 1
        tp, fp, fn = int(np.sum(chosen & positive)), int(np.sum(chosen & ~positive)), int(np.sum(~chosen & positive))
        output["f1"] = 2 * tp / max(1, 2 * tp + fp + fn)
        if positive.any() and (~positive).any():
            from scipy.stats import rankdata
            ranks = rankdata(scores[:, 1] - scores[:, 0])
            n1, n0 = int(positive.sum()), int((~positive).sum())
            output["auc"] = float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        else:
            output["auc"] = None
    return output

