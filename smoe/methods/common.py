"""Helpers shared by the per-method training loops."""
import contextlib

import numpy as np
import torch
import torch.nn.functional as F


def autocast(device):
    """fp16 autocast on CUDA (as in the original runs); a no-op on CPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def probs_from_logits(pred):
    """Sigmoid for a single logit per sample, softmax for two."""
    if pred.shape[1] == 1:
        return torch.sigmoid(pred.detach().squeeze(1))
    return F.softmax(pred.detach(), dim=1)


class Collector:
    """Accumulates per-batch losses, probabilities, labels and attributes for one epoch."""

    def __init__(self):
        self.losses, self.probs, self.gts, self.attrs = [], [], [], []

    def add(self, loss, probs, target, attr=None):
        self.losses.append(loss)
        self.probs.append(probs.float().cpu().numpy())
        self.gts.append(target.detach().cpu().numpy())
        if attr is not None:
            self.attrs.append(torch.vstack(attr).cpu().numpy())

    def result(self):
        out = [float(np.mean(self.losses)), np.concatenate(self.probs), np.concatenate(self.gts)]
        if self.attrs:
            out.append(np.concatenate(self.attrs, axis=1).astype(int))
        return out
