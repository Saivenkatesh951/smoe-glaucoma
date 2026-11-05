# From: https://raw.githubusercontent.com/facebookresearch/mae/main/models_vit.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial  # Importing partial function for creating partial functions that allows you 
# to create new functions with some arguments pre-filled.
import torch  # Importing the main PyTorch library
import torch.nn as nn  # Importing the neural network module from PyTorch
import math  #Importing the math module for mathematical functions
import json  #Importing the JSON module for handling JSON data
import numpy as np  #Importing NumPy for numerical operations
# from torch._six import inf  # (Commented out) Importing infinity from PyTorch's internal six module
from torch import inf  #Importing infinity from the torch module
import timm.models.vision_transformer  # Importing the Vision Transformer model from the timm library

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=True, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        #self.global_pool = global_pool
        if global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        global_pool = True
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
        
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        #x = self.pool(x)
        #x = self.fc_norm(x)
        #x = self.head_drop(x)
        return x if pre_logits else self.head(x)


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# Class for managing mixed precision training with gradient scaling and clipping in PyTorch.
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph) # Scale the loss and compute the gradients.

        if update_grad:
            if clip_grad is not None:
                assert parameters is not None 
                self._scaler.unscale_(optimizer)  #unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad) 
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)

            self._scaler.step(optimizer) # Update the optimizer's internal state based on the scaled loss.
            self._scaler.update() # Update the scale factor for the next iteration.

        else:
            norm = None

        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

# Function to calculate the gradient norm of a given set of parameters.
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    # Filter out parameters that don't have a gradient
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type) 
    # If there are no parameters with a gradient, return a tensor with a value of 0.
    if len(parameters) == 0:
        return torch.tensor(0.)
    # Get the device of the first parameter's gradient
    device = parameters[0].grad.device
    # If the norm type is infinity, find the maximum absolute value of the gradient among all parameters.
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else: # If the norm type is not infinity, calculate the norm of the gradients.
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

"""Adjusts the learning rate with a warmup phase followed by cosine decay."""
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))) 
         
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr 

#Function to create parameter groups for layer-wise learning rate decay.
def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """ 
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
             
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print(param_groups.keys())
    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    # If the parameter name is in the list of special tokens, assign it to layer 0.
    if name in ['cls_token', 'pos_embed', 'mask_token', 'register_tokens']: #add mask_token, register_tokens for DINOv2
        return 0
    # If the parameter name starts with 'patch_embed', assign it to layer 0.
    elif name.startswith('patch_embed'):
        return 0
    # If the parameter name starts with 'blocks', assign it to the layer number specified in the name.
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers
    
# Interpolates position embeddings from a checkpoint to match the current model's patch size.
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # Reshape the position tokens to the original size and permute the dimensions.
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            # Interpolate the position tokens to the new size using bicubic interpolation.
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # Permute the dimensions of the interpolated position tokens and flatten them.
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            # Concatenate the extra tokens with the interpolated position tokens.
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed