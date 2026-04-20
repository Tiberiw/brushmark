import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torchvision.models.resnet import ResNet

def get_resnet50_model(n_classes: int, device: torch.device, dropout: float = 0.5):
    """Return a ResNet50 with frozen backbone and a trainable classification head."""
    model = resnet50(weights = ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential( # type: ignore[assignment]
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(512, n_classes)
    )
    model = model.to(device)
    return model


def get_resnet18_model(n_classes: int, device: torch.device, dropout: float = 0.5):
    """Return a ResNet18 with frozen backbone and a trainable classification head."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential( # type: ignore[assignment]
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(512, n_classes)
    )
    model = model.to(device)
    return model

def get_resnet_model(
        model: str,
        dropout: float,
        n_classes: int,
        device: torch.device
        ) -> ResNet:
    if model == "resnet50":
        return get_resnet50_model(n_classes, device, dropout)
    if model == "resnet18":
        return get_resnet18_model(n_classes, device, dropout)
    return get_resnet18_model(n_classes, device, dropout)
    