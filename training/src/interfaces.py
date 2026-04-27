from dataclasses import dataclass, field
from typing import Any

@dataclass
class ParamGroup:
    layer_name: str
    lr: float

@dataclass(frozen=True)
class OptimizerConfig:
    weight_decay: float
    param_groups: list[ParamGroup]

@dataclass
class Warmup:
    enabled: bool
    epochs: int
    optimizer: OptimizerConfig

@dataclass
class LossFunctionConfig:
    label_smoothing: float
    sampling_weight_scale: float

@dataclass
class Training:
    epochs: int
    loss_fn: LossFunctionConfig
    optimizer: OptimizerConfig

@dataclass
class TransformComposeConfig:
    _target_: str = "torchvision.transforms.Compose"
    transforms: list[Any] = field(default_factory=list)

@dataclass
class LoaderConfig:
    batch_size: int
    num_workers: int

@dataclass
class DataConfig:
    n_classes: int
    valid_size: float
    dataloader: LoaderConfig
    train_transforms: TransformComposeConfig
    val_transforms: TransformComposeConfig

@dataclass
class ModelConfig:
    name: str
    dropout: float

@dataclass
class Configuration:
    experiment_name: str
    warmup: Warmup
    training: Training
    model: ModelConfig
    data: DataConfig