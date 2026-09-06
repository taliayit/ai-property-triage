"""Two-head transfer-learning model: a frozen, ImageNet-pretrained
EfficientNet-B0 backbone feeding two independent linear heads - one for
room_type (6-way), one for condition_score (5-way, scores 1-5 mapped to
class indices 0-4).

EfficientNet-B0 (5.3M params) was chosen over ResNet-50 (25M params)
mainly for CPU training/inference speed here - with only 154 training
images, a larger backbone wouldn't out-learn a smaller one anyway (the
backbone is frozen; only the two heads ever get gradients).
"""

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from config import CONDITION_SCORES, ROOM_TYPES


def build_transform():
    """The exact preprocessing (resize/crop/normalize) the pretrained
    backbone's own ImageNet weights expect - used identically for
    inference and for the non-augmented eval-time dataset transform."""
    return EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()


class RoomConditionNet(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features  # 1280

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=0.3)
        self.room_head = nn.Linear(in_features, len(ROOM_TYPES))
        self.condition_head = nn.Linear(in_features, len(CONDITION_SCORES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.room_head(x), self.condition_head(x)


def load_model(checkpoint_path) -> RoomConditionNet:
    model = RoomConditionNet(freeze_backbone=True)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
