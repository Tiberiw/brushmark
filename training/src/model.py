import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50_model(n_classes: int, device: torch.device):
    model = resnet50(weights = ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential( # type: ignore[assignment]
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, n_classes)
    )
    model = model.to(device)
    return model

