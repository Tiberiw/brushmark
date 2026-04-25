from dataclasses import dataclass, field

@dataclass
class ParamGroup:
    layer_name: str
    lr: float

@dataclass(frozen=True)
class Optimizer:
    name: str
    weight_decay: float
    param_groups: list[ParamGroup]

@dataclass
class Warmup:
    enabled: bool
    epochs: int
    optimizer: Optimizer

@dataclass
class LossFunction:
    label_smoothing: float
    sampling_weight_scale: float

@dataclass
class Training:
    epochs: int
    batch_size: int
    loss_fn: LossFunction
    num_workers: int
    optimizer: Optimizer

@dataclass
class Model:
    name: str
    dropout: float

@dataclass
class Configuration:
    experiment_name: str
    warmup: Warmup
    training: Training
    model: Model