"""Normal multi-modal baseline and the OPM / OGM / OPM+OGM balancing methods (paper Table 2).

normal   late fusion, no balancing
opm      On-the-fly Prediction Modulation: randomly drop the dominant modality's features
ogm      On-the-fly Gradient Modulation (with Gaussian Enhancement): scale down the
         dominant encoder's gradients and add Gaussian noise
opm_ogm  both
"""
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from ..data import MODALITIES, channels
from ..models.fusion import FusionModel
from .common import Collector, autocast, probs_from_logits


def _in_channels(args):
    return tuple(channels(m) for m in MODALITIES[args.dataset])


def build(args, device):
    model = FusionModel(
        _in_channels(args), use_opm=args.method in ("opm", "opm_ogm"),
        p_exe=args.p_exe, q_base=args.q_base, lam=args.lam, device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
    return SimpleNamespace(model=model.to(device), optimizer=optimizer, scheduler=scheduler, criterion=nn.CrossEntropyLoss())


def _modulate_gradients(model, performance_1, performance_2, epoch, args):
    """OGM: shrink the conv-weight gradients of the encoder with the higher performance."""
    tanh, relu = nn.Tanh(), nn.ReLU(inplace=True)
    ratio_1 = performance_1 / performance_2
    ratio_2 = performance_2 / performance_1
    if ratio_1 > 1:
        coeff_1, coeff_2 = 1 - tanh(args.alpha * relu(ratio_1)), 1
    else:
        coeff_1, coeff_2 = 1, 1 - tanh(args.alpha * relu(ratio_2))

    if not args.modulation_starts <= epoch <= args.modulation_ends:
        return
    for name, parms in model.named_parameters():
        layer = name.split(".")[0]
        if layer not in ("encoder_1", "encoder_2") or parms.grad is None or parms.grad.dim() != 4:
            continue
        coeff = coeff_1 if layer == "encoder_1" else coeff_2
        if args.modulation == "OGM_GE":
            parms.grad = parms.grad * coeff + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
        else:
            parms.grad *= coeff


def train_one_epoch(state, loader, epoch, args, device):
    use_opm = args.method in ("opm", "opm_ogm")
    use_ogm = args.method in ("ogm", "opm_ogm")
    state.model.train()
    col = Collector()

    for inputs, target, _ in loader:
        with autocast(device):
            image1, image2 = inputs[0].to(device), inputs[1].to(device)
            target = target.to(device).long()
            state.model.zero_grad()

            _, _, pred, update_flag, performance_1, performance_2 = state.model(image1.float(), image2.float(), target, warm_up=0)

            if use_opm:
                if torch.sum(update_flag, dim=0) == 0:
                    continue
                target = target[update_flag != 0]

            loss = state.criterion(pred, target)
            loss.backward()
            col.add(loss.item(), probs_from_logits(pred), target)

        if use_ogm:
            _modulate_gradients(state.model, performance_1, performance_2, epoch, args)
        state.optimizer.step()

    return col.result()


@torch.no_grad()
def evaluate(state, loader, epoch, args, device):
    state.model.eval()
    col = Collector()

    for inputs, target, attr in loader:
        image1, image2 = inputs[0].to(device), inputs[1].to(device)
        target = target.to(device).long()
        _, _, pred, _, _, _ = state.model(image1.float(), image2.float(), target, warm_up=1)
        col.add(state.criterion(pred, target).item(), probs_from_logits(pred), target, attr)

    return col.result()
