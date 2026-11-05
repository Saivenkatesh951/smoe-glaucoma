"""ImageNet-pretrained CNN backbones used by the baselines and as experts/encoders."""
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B1_Weights,
    ResNet50_Weights,
    densenet121,
    efficientnet_b1,
    resnet50,
)

CNN_MODELS = ("densenet", "efficientnet", "resnet")


def _channel_adapter(in_dim, first_layer):
    """1x1 conv mapping in_dim channels to 3, followed by the pretrained first layer."""
    return nn.Sequential(nn.Conv2d(in_dim, 3, kernel_size=1, stride=1), nn.GELU(), first_layer)


def create_model(model_type="efficientnet", in_dim=3, out_dim=1, use_pretrained=True):
    """Classifier with an ImageNet backbone and a fresh out_dim head."""
    if model_type == "efficientnet":
        model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2 if use_pretrained else None)
        if in_dim != 3:
            model.features[0][0] = _channel_adapter(in_dim, model.features[0][0])
        model.classifier[1] = nn.Linear(1280, out_dim, bias=False)

    elif model_type == "resnet":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None)
        if in_dim != 3:
            model.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(2048, out_dim, bias=True)

    elif model_type == "densenet":
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else None)
        if in_dim != 3:
            model.features.conv0 = _channel_adapter(in_dim, model.features.conv0)
        model.classifier = nn.Linear(1024, out_dim, bias=True)

    else:
        raise ValueError(f"model_type must be one of {CNN_MODELS}, got '{model_type}'")

    return model


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B1 feature extractor returning [B, 1280, h, w] feature maps."""

    def __init__(self, in_channels=3):
        super().__init__()
        backbone = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
        if in_channels != 3:
            backbone.features[0][0] = _channel_adapter(in_channels, backbone.features[0][0])
        self.encoder = backbone.features

    def forward(self, x):
        return self.encoder(x)
