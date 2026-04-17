import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FASModel(nn.Module):
    def __init__(self, pretrained=True):
        super(FASModel, self).__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)

        # Replace the final fully connected layer for binary classification
        # 0: Real, 1: Spoof
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)

def get_device():
    """Returns the available device: GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
