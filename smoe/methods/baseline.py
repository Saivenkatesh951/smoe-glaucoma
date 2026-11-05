"""Uni-modal baselines (paper Table 1): DenseNet-121, EfficientNet-B1, ResNet-50, ViT-B/16."""
from types import SimpleNamespace

import timm
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from torch.optim.lr_scheduler import StepLR

from ..data import channels
from ..models import models_vit
from ..models.backbones import create_model
from .common import Collector, autocast, probs_from_logits


def _build_vit(args, in_dim):
    model = models_vit.vit_base_patch16(num_classes=1, drop_path_rate=args.drop_path, global_pool=True)

    if args.vit_weights != "scratch":
        if args.vit_weights == "imagenet":
            print("Load ImageNet pre-trained checkpoint from timm")
            imagenet_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
            checkpoint_model = imagenet_model.state_dict()
            del imagenet_model
        else:  # path to an MAE-style checkpoint with a 'model' entry
            print(f"Load pre-trained checkpoint from: {args.vit_weights}")
            checkpoint_model = torch.load(args.vit_weights, map_location="cpu")["model"]

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        models_vit.interpolate_pos_embed(model, checkpoint_model)
        model.load_state_dict(checkpoint_model, strict=False)

    trunc_normal_(model.head.weight, std=2e-5)

    if in_dim != 3:
        model.patch_embed.proj = nn.Sequential(nn.Conv2d(in_dim, 3, kernel_size=1, stride=1), nn.GELU(), model.patch_embed.proj)
    return model


def build(args, device):
    in_dim = channels(args.modality)

    if args.model_type == "ViT-B":
        model = _build_vit(args, in_dim)
        param_groups = models_vit.param_groups_lrd(
            model, args.weight_decay, no_weight_decay_list=model.no_weight_decay(), layer_decay=args.layer_decay
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        scaler = models_vit.NativeScalerWithGradNormCount()
        scheduler = None  # per-iteration cosine schedule, see train_one_epoch
    else:
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.0, 0.1), weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    return SimpleNamespace(
        model=model.to(device), optimizer=optimizer, scheduler=scheduler, scaler=scaler, criterion=nn.BCEWithLogitsLoss()
    )


def train_one_epoch(state, loader, epoch, args, device):
    state.model.train()
    col = Collector()

    for i, (inputs, target, _) in enumerate(loader):
        if args.model_type == "ViT-B":
            lr = models_vit.adjust_learning_rate(state.optimizer, i / len(loader) + epoch, args)
            if i == 0:
                print(f"lr: {lr}")

        with autocast(device):
            x = inputs[0].to(device)
            target = target.to(device)
            pred = state.model(x)
            loss = state.criterion(pred.squeeze(1), target)

        col.add(loss.item(), probs_from_logits(pred), target)

        if args.model_type == "ViT-B":
            state.scaler(loss, state.optimizer, clip_grad=None, parameters=state.model.parameters(), create_graph=False, update_grad=True)
        else:
            state.scaler.scale(loss).backward()
            state.scaler.step(state.optimizer)
            state.scaler.update()
        state.optimizer.zero_grad()

    return col.result()


@torch.no_grad()
def evaluate(state, loader, epoch, args, device):
    state.model.eval()
    col = Collector()

    for inputs, target, attr in loader:
        x = inputs[0].to(device)
        target = target.to(device)
        pred = state.model(x)
        loss = state.criterion(pred.squeeze(1), target)
        col.add(loss.item(), probs_from_logits(pred), target, attr)

    return col.result()
