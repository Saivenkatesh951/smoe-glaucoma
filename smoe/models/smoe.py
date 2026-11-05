"""Balanced Soft Mixture-of-Experts (SMoE), the method proposed in the paper.

Three experts share the input: two uni-modal EfficientNet-B1 classifiers (one per modality)
and one multi-modal late-fusion model (no modulation). A small CNN gate looks at both images and produces
softmax weights over the experts; the prediction is the weighted sum of expert logits.
Training adds a coefficient-of-variation load-balancing loss on the gate weights (Eq. 1).
The design is inspired by DynMM (Xue & Marculescu, CVPR-W 2023).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import create_model
from .fusion import FusionModel


def diff_softmax(logits, tau=1.0, hard=False, dim=-1):
    """Softmax with an optional straight-through hard (one-hot) assignment."""
    y_soft = (logits / tau).softmax(dim)
    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class Gate(nn.Module):
    """Two conv-BN-Tanh blocks over the channel-concatenated inputs, then 1x1 conv to expert logits."""

    def __init__(self, in_channels, branch_num=3, hidden_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=5, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh(),
        )
        self.fc = nn.Conv2d(hidden_dim, branch_num, kernel_size=1, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(self.conv(x), 1)
        return self.fc(y).squeeze(-1).squeeze(-1)


def load_balancing_loss(gating_weights, lamda_dyn):
    """lambda_dyn * CV^2 of the per-expert gate load summed over the batch (paper Eq. 1)."""
    expert_load = gating_weights.sum(dim=0)
    cv = expert_load.std() / (expert_load.mean() + 1e-8)
    return lamda_dyn * cv ** 2


class SMoE(nn.Module):
    def __init__(self, in_channels=(3, 3), device="cuda"):
        super().__init__()
        self.branch1 = create_model("efficientnet", in_dim=in_channels[0], out_dim=2)  # modality 1 expert
        self.branch2 = create_model("efficientnet", in_dim=in_channels[1], out_dim=2)  # modality 2 expert
        self.branch3 = FusionModel(in_channels, use_opm=False, device=device)  # multi-modal expert
        self.gate = Gate(in_channels=sum(in_channels), branch_num=3, hidden_dim=32)
        self.temp = 1
        self.hard_gate = False
        self.weight_list = torch.Tensor()
        self.store_weight = False

    def reset_weight(self):
        """Start recording gate weights (called before evaluation)."""
        self.weight_list = torch.Tensor()
        self.store_weight = True

    def weight_stat(self):
        """Print the mean gate weight of each expert over the recorded samples."""
        mean = torch.mean(self.weight_list, dim=0)
        print(f"mean branch weight {mean[0].item():.4f}, {mean[1].item():.4f}, {mean[2].item():.4f}")
        self.store_weight = False
        return mean

    def forward(self, inputs, target):
        weight = diff_softmax(self.gate(torch.cat(inputs, dim=1)), tau=self.temp, hard=self.hard_gate)
        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.detach().cpu()))

        # Experts run in this order so random dropout draws match the original code
        pred1 = self.branch1(inputs[0])
        pred2 = self.branch2(inputs[1])
        pred3 = self.branch3(inputs[0], inputs[1], target)[2]  # fused logits
        output = weight[:, 0:1] * pred1 + weight[:, 1:2] * pred2 + weight[:, 2:3] * pred3
        return output, weight
