"""Two-encoder late-fusion models used by Normal, OPM, OGM, OPM+OGM, CGGM and SMoE.

OPM (On-the-fly Prediction Modulation) and OGM-GE (On-the-fly Gradient Modulation with
Gaussian Enhancement) follow Wei et al. (TPAMI 2024) and Peng et al. (CVPR 2022).
CGGM (Classifier-Guided Gradient Modulation) follows Guo et al. (NeurIPS 2024).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import EfficientNetEncoder

FEATURE_DIM = 1280  # EfficientNet-B1 feature size


class ConcatFusion(nn.Module):
    def __init__(self, in_c_x, in_c_y, out_c):
        super().__init__()
        self.fxy = nn.Linear(in_c_x + in_c_y, out_c)

    def forward(self, x, y):
        return x, y, self.fxy(torch.cat([x, y], dim=1))


class _RescaleByTheta(torch.autograd.Function):
    """Rescales kept features by 1 / (1 - theta) after modality drop."""

    @staticmethod
    def forward(ctx, input, theta):
        ctx.save_for_backward(input, theta)
        return input / (1 - theta.item())

    @staticmethod
    def backward(ctx, grad_output):
        input, theta = ctx.saved_tensors
        return 1 / (1 - theta.item()) * grad_output.clone(), theta


class ModalityDrop:
    """OPM: drop each modality's features with probability q (computed by `calcu_q`)."""

    def __init__(self, dim_list, p_exe=0.7, device="cuda"):
        self.dim_list = dim_list
        self.p_exe = p_exe
        self.device = device

    def execute_drop(self, feat_list, q):
        B, D = feat_list[0].shape
        exe_drop = torch.tensor(np.random.rand(1)).to(device=self.device) >= 1 - self.p_exe
        if not exe_drop:
            return feat_list, torch.ones([B], dtype=torch.int32, device=self.device)

        num_mod = len(feat_list)
        theta = sum(self.dim_list * q) / sum(self.dim_list)

        mask = torch.distributions.Bernoulli(1 - q).sample([B, 1]).permute(2, 1, 0).contiguous()
        mask = mask.reshape(num_mod, B, -1).to(device=self.device)  # [M, B, 1]
        feats = torch.mul(torch.stack(feat_list, dim=0), mask)  # [M, B, D]
        feats = _RescaleByTheta.apply(feats, theta)
        mask = torch.transpose(mask, 0, 1).squeeze(-1)  # [B, M]
        update_flag = torch.sum(mask, dim=1) > 0
        cleaned = torch.masked_select(feats, update_flag.unsqueeze(-1)).reshape(num_mod, -1, D)
        cleaned = [c.squeeze(0) for c in torch.chunk(cleaned, num_mod, dim=0)]
        return cleaned, update_flag


def calcu_q(performance_1, performance_2, q_base, fix_lambda):
    """OPM drop probability of each modality from its discriminative discrepancy ratio."""
    q = torch.tensor([0.0, 0.0])
    relu = nn.ReLU(inplace=True)
    ratio_1 = torch.tanh(relu(performance_1 / performance_2 - 1))
    ratio_2 = torch.tanh(relu(performance_2 / performance_1 - 1))
    q[0] = q_base * (1 + fix_lambda * ratio_1) if ratio_1 > 0 else 0
    q[1] = q_base * (1 + fix_lambda * ratio_2) if ratio_2 > 0 else 0
    return torch.clip(q, 0.0, 1.0)


def _pooled(encoder, x):
    return F.adaptive_avg_pool2d(encoder(x), 1).flatten(1)  # [B, 1280]


class FusionModel(nn.Module):
    """Two EfficientNet-B1 encoders + concat fusion straight to 2 logits.

    Also computes each modality's uni-modal logits (t1, t2) from its half of the fusion
    weights; OPM and OGM use their summed correct-class probabilities as "performance".
    """

    def __init__(self, in_channels=(3, 3), use_opm=False, p_exe=0.7, q_base=0.5, lam=0.5, device="cuda"):
        super().__init__()
        self.device = device
        self.use_opm = use_opm
        self.q_base = q_base
        self.lam = lam
        self.encoder_1 = EfficientNetEncoder(in_channels=in_channels[0])
        self.encoder_2 = EfficientNetEncoder(in_channels=in_channels[1])
        self.fusion_model = ConcatFusion(FEATURE_DIM, FEATURE_DIM, 2)
        if use_opm:
            self.modality_drop = ModalityDrop(torch.tensor([FEATURE_DIM, FEATURE_DIM]), p_exe=p_exe, device=device)

    def forward(self, mod_1, mod_2, label, warm_up=0):
        out_1 = _pooled(self.encoder_1, mod_1)
        out_2 = _pooled(self.encoder_2, mod_2)
        B = out_1.shape[0]

        w = self.fusion_model.fxy.weight.clone().detach()
        b = self.fusion_model.fxy.bias.clone().detach()
        t1 = torch.mm(out_1, w[:, :FEATURE_DIM].T) + b / 2
        t2 = torch.mm(out_2, w[:, FEATURE_DIM:].T) + b / 2

        # Sum of correct-class probabilities over the batch
        label = label.long().view(-1, 1)
        performance_1 = F.softmax(t1, dim=1).gather(1, label).sum()
        performance_2 = F.softmax(t2, dim=1).gather(1, label).sum()

        if warm_up == 0 and self.use_opm:
            self.q = calcu_q(performance_1, performance_2, self.q_base, fix_lambda=self.lam)
            cleaned, update_flag = self.modality_drop.execute_drop([out_1, out_2], self.q)
            _, _, out = self.fusion_model(cleaned[0], cleaned[1])
            return t1, t2, out, update_flag, performance_1, performance_2

        _, _, out = self.fusion_model(out_1, out_2)
        return t1, t2, out, torch.ones([B], dtype=torch.int32, device=self.device), performance_1, performance_2


class CGGMFusionModel(nn.Module):
    """Two EfficientNet-B1 encoders + concat fusion to 1280-d + linear head (CGGM setup).

    Also returns the detached uni-modal features `hs` that the CGGM classifiers consume.
    """

    def __init__(self, in_channels=(3, 3)):
        super().__init__()
        self.encoder_1 = EfficientNetEncoder(in_channels=in_channels[0])
        self.encoder_2 = EfficientNetEncoder(in_channels=in_channels[1])
        self.fusion_model = ConcatFusion(FEATURE_DIM, FEATURE_DIM, FEATURE_DIM)
        self.out_layer = nn.Linear(FEATURE_DIM, 2)

    def forward(self, mod_1, mod_2):
        out_1 = _pooled(self.encoder_1, mod_1)
        out_2 = _pooled(self.encoder_2, mod_2)
        hs = [out_1.clone().detach(), out_2.clone().detach()]
        _, _, fused = self.fusion_model(out_1, out_2)
        return self.out_layer(fused), hs


class ClassifierGuided(nn.Module):
    """One linear classifier per modality on the detached uni-modal features."""

    def __init__(self, output_dim=2, num_mod=2, in_dim=FEATURE_DIM):
        super().__init__()
        self.classifiers = nn.ModuleList([_LinearClassifier(in_dim, output_dim) for _ in range(num_mod)])

    def cal_coeff(self, y, cls_res):
        """Training accuracy of each uni-modal classifier on the batch."""
        return [float((r.view(-1, 2).argmax(dim=1) == y.view(-1)).float().mean()) for r in cls_res]

    def forward(self, hs):
        return [clf(h) for clf, h in zip(self.classifiers, hs)]


class _LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.out_layer(x)


def cal_cos(cls_weights, fusion_weight):
    """Cosine similarity between each uni-modal classifier's weights and the fusion head's.

    As in the original CGGM code used for the paper, this compares the layers' weights
    (not their gradients).
    """
    fgn = fusion_weight.clone().view(-1)
    return [F.cosine_similarity(w.clone().view(-1), fgn, dim=0) for w in cls_weights]
