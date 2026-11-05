"""Classifier-Guided Gradient Modulation (CGGM, Guo et al., NeurIPS 2024) (paper Table 2).

A linear classifier per modality is trained on the detached uni-modal features. The change
in each classifier's accuracy sets per-encoder gradient coefficients, and a direction term
(l_gm) aligning the classifiers with the fusion head is added to the next batch's loss.
"""
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from ..data import MODALITIES, channels
from ..models.fusion import CGGMFusionModel, ClassifierGuided, cal_cos
from .common import Collector, probs_from_logits

NUM_MOD = 2


def build(args, device):
    model = CGGMFusionModel(tuple(channels(m) for m in MODALITIES[args.dataset]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    classifier = ClassifierGuided(output_dim=2, num_mod=NUM_MOD, in_dim=1280)
    cls_optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.cls_lr, weight_decay=3e-3)
    return SimpleNamespace(
        model=model.to(device), optimizer=optimizer, scheduler=StepLR(optimizer, step_size=6, gamma=0.1),
        classifier=classifier.to(device), cls_optimizer=cls_optimizer, criterion=nn.CrossEntropyLoss(),
    )


def train_one_epoch(state, loader, epoch, args, device):
    model, classifier, criterion = state.model, state.classifier, state.criterion
    model.train()
    col = Collector()
    acc1 = [0] * NUM_MOD
    l_gm = None

    for inputs, target, _ in loader:
        image1, image2 = inputs[0].to(device), inputs[1].to(device)
        target = target.to(device).long()
        model.zero_grad()

        pred, hs = model(image1.float(), image2.float())
        loss = criterion(pred, target)
        if l_gm is not None:
            loss += args.lamda * l_gm
        loss.backward()

        state.cls_optimizer.zero_grad()
        cls_res = [r.view(-1, 2) for r in classifier(hs)]
        fusion_weight = model.out_layer.weight
        cls_loss = sum(criterion(r, target) for r in cls_res)
        cls_loss.backward()
        cls_weights = [clf.out_layer.weight for clf in classifier.classifiers]

        llist = cal_cos(cls_weights, fusion_weight)
        acc2 = classifier.cal_coeff(target, cls_res)
        diff = [acc2[i] - acc1[i] for i in range(NUM_MOD)]
        diff_sum = sum(diff)
        acc1 = acc2

        if abs(diff_sum) <= 1e-8:
            l_gm = 0.0
        else:
            coeff = [(diff_sum - d) / diff_sum for d in diff]
            l_gm = (np.sum(np.abs(coeff)) - (coeff[0] * llist[0] + coeff[1] * llist[1])) / NUM_MOD
            for name, params in model.named_parameters():
                if params.grad is None:
                    continue
                if "encoder_1" in name:
                    params.grad *= coeff[0] * args.rou
                if "encoder_2" in name:
                    params.grad *= coeff[1] * args.rou

        state.cls_optimizer.step()
        col.add(loss.item(), probs_from_logits(pred), target)
        state.optimizer.step()

    return col.result()


@torch.no_grad()
def evaluate(state, loader, epoch, args, device):
    state.model.eval()
    col = Collector()

    for inputs, target, attr in loader:
        image1, image2 = inputs[0].to(device), inputs[1].to(device)
        target = target.to(device).long()
        pred, _ = state.model(image1.float(), image2.float())
        col.add(state.criterion(pred, target).item(), probs_from_logits(pred), target, attr)

    return col.result()
