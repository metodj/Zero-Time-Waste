import torch
import numpy as np
from typing import Dict, Optional, List


def probs_decrease(probs: np.array) -> np.array:
    L = len(probs)
    diffs = []
    for i in range(L):
        for j in range(i + 1, L):
            diffs.append(probs[j] - probs[i])
    return np.array(diffs)


def modal_probs_decreasing(
    _preds: Dict[int, torch.Tensor],
    _probs: torch.Tensor,
    layer: Optional[int] = None,
    verbose: bool = False,
    N: int = 10000,
    diffs_type: str = "consecutive",
    thresholds: List[float] = [-0.01, -0.05, -0.1, -0.2, -0.5],
    return_ids: bool = False
) -> Dict[float, float]:
    """
    nr. of decreasing modal probability vectors in anytime-prediction regime
    function can also be used for ground truth probabilities, set layer=None
    """
    nr_non_decreasing = {threshold: 0 for threshold in thresholds}
    diffs = {threshold: [] for threshold in thresholds}
    for i in range(N):
        if layer is None:
            c = _preds[i]
        else:
            c = _preds[layer - 1][i]
        probs_i = _probs[:, i, c].cpu().numpy()
        if diffs_type == "consecutive":
            diffs_i = np.diff(probs_i)
        elif diffs_type == "all":
            diffs_i = probs_decrease(probs_i)
        else:
            raise ValueError()
        # diffs.append(diffs_i.min())
        for threshold in nr_non_decreasing.keys():
            if np.all(diffs_i >= threshold):
                nr_non_decreasing[threshold] += 1
            else:
                diffs[threshold].append(i)
                if verbose:
                    print(i, probs_i)
    # print(nr_non_decreasing)
    # print(np.mean(diffs))
    nr_decreasing = {
        -1.0 * k: ((N - v) / N) * 100 for k, v in nr_non_decreasing.items()
    }
    if return_ids:
        return nr_decreasing, diffs
    else:
        return nr_decreasing
    

def f_probs_ovr_poe_logits_weighted_generalized(logits, threshold=0.0, weights=None):
    L, C = logits.shape[0], logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.0
    if weights is not None:
        assert logits.shape[0] == weights.shape[0]
        for l in range(L):
            probs[l, :, :] = probs[l, :, :] ** weights[l]
    probs = np.cumprod(probs, axis=0)
    # normalize
    probs = probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2)
    return probs



def f_probs_ovr_poe_logits_weighted_generalized_hetero_weights_per_ee(logits: torch.Tensor, weights: Dict[int, torch.Tensor], threshold: float = 0.0) -> torch.Tensor:
    L, C = logits.shape[0], logits.shape[-1]
    probs = logits.numpy().copy()
    probs[probs < threshold] = 0.0
    probs_prod = []
    for l in range(L):
        probs_l = probs[:l + 1, :, :].copy()
        for i in range(l + 1):
            probs_l[i, :, :] = probs[i, :, :] ** weights[l][i]
        probs_l = np.cumprod(probs_l, axis=0)
        probs_prod.append(probs_l[-1, :, :])
    probs = np.stack(probs_prod, axis=0)
    # normalize
    probs = probs / np.repeat(probs.sum(axis=2)[:, :, np.newaxis], C, axis=2)
    return probs


def anytime_caching(_probs: torch.Tensor, N: int, L: int) -> torch.Tensor:
    _preds = []
    _probs_stateful = []
    for n in range(N):
        preds_all, probs_stateful_all = [], []
        max_prob_all, pred_all, max_id = 0., None, 0.
        for l in range(L):
            _max_prob, _pred = _probs[l, n, :].max(), _probs[l, n, :].argmax()
            if _max_prob >= max_prob_all:
                max_prob_all = _max_prob
                pred_all = _pred
                prob_stateful_all = _probs[l, n, :]
                max_id = l
            else:
                prob_stateful_all = _probs[max_id, n, :]
            preds_all.append(pred_all)
            probs_stateful_all.append(prob_stateful_all)
        _preds.append(torch.stack(preds_all))
        _probs_stateful.append(torch.stack(probs_stateful_all))

    _preds = torch.stack(_preds)
    _probs_stateful = torch.stack(_probs_stateful)
    return _probs_stateful.permute(1, 0, 2)