"""Balanced Soft Mixture-of-Experts, the proposed method (paper Table 2, Section 4.2).

loss = cross-entropy + lambda_dyn * CV(expert load)^2   (paper Eq. 1)
"""
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from ..data import MODALITIES, channels
from ..models.smoe import SMoE, load_balancing_loss
from .common import Collector, probs_from_logits


def build(args, device):
    model = SMoE(tuple(channels(m) for m in MODALITIES[args.dataset]), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return SimpleNamespace(
        model=model.to(device), optimizer=optimizer, scheduler=StepLR(optimizer, step_size=6, gamma=0.1),
        criterion=nn.CrossEntropyLoss(),
    )


def _loss(state, inputs, target, args):
    pred, gating_weights = state.model(inputs, target)
    return pred, state.criterion(pred, target) + load_balancing_loss(gating_weights, args.lamda_dyn)


def train_one_epoch(state, loader, epoch, args, device):
    state.model.train()
    col = Collector()

    for inputs, target, _ in loader:
        inputs = [x.to(device).float() for x in inputs]
        target = target.to(device).long()
        state.optimizer.zero_grad()

        pred, loss = _loss(state, inputs, target, args)
        loss.backward()
        col.add(loss.item(), probs_from_logits(pred), target)
        state.optimizer.step()

    return col.result()


@torch.no_grad()
def evaluate(state, loader, epoch, args, device):
    state.model.eval()
    state.model.reset_weight()
    col = Collector()

    for inputs, target, attr in loader:
        inputs = [x.to(device).float() for x in inputs]
        target = target.to(device).long()
        pred, loss = _loss(state, inputs, target, args)
        col.add(loss.item(), probs_from_logits(pred), target, attr)

    state.model.weight_stat()
    return col.result()
