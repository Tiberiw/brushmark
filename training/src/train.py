import torch
import torch.nn as nn
from data_setup import create_dataloaders
from download_data import setup_data
from model import get_resnet_model
from engine import train
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet
import hydra
from hydra.core.hydra_config import HydraConfig
from interfaces import Configuration, LossFunctionConfig, OptimizerConfig
from omegaconf import OmegaConf

def set_seeds(seed: int = 42):
    """ Set seeds for random operations
        Args:
            seed: int - Random seed for operations
    """
    torch.manual_seed(seed) # General torch operations
    torch.cuda.manual_seed(seed) # CUDA torch operation on gpu

def get_optimizer(cfg: OptimizerConfig, model: nn.Module):
    weight_decay = cfg.weight_decay
    param_groups = []
    for param_group in cfg.param_groups:
        submodule = model.get_submodule(param_group.layer_name)
        layer_params = list(submodule.parameters())
        for p in layer_params:
            p.requires_grad = True
        param_groups.append({ "params": layer_params, "lr": param_group.lr })
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

def get_loss_fn(cfg: LossFunctionConfig, class_weights: dict[int, float], n_classes: int, device: torch.device) -> nn.Module:
    # penalize minority errors more, but not as extreme as linear (sqrt dampening)
    weights = torch.tensor(
        [class_weights[i] ** cfg.sampling_weight_scale for i in range(n_classes)],
        dtype=torch.float,
        device=device,
    )
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

def run_complete_training(
        cfg: Configuration,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        model: ResNet,
        loss_fn: nn.Module,
        device: torch.device,
        n_classes: int,
        idx_to_class: dict[int, str],
        current_run_folder: Path,
        writer: SummaryWriter
    ) -> None: 
    if cfg.warmup.enabled:
        optimizer = get_optimizer(cfg.warmup.optimizer, model)
        train(cfg.warmup.epochs, train_loader, valid_loader, model, optimizer, loss_fn,
            device, n_classes, idx_to_class, current_run_folder)
        
    optimizer = get_optimizer(cfg.training.optimizer, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )
    train(cfg.training.epochs, train_loader, valid_loader, model, optimizer, loss_fn,
        device, n_classes, idx_to_class, current_run_folder, writer, scheduler, True)
    
OmegaConf.register_new_resolver(
    "choice",
    lambda key: HydraConfig.get().runtime.choices[key],
)

@hydra.main(version_base="1.2", config_path="../configs", config_name="loaders_multirun")
def main(cfg: Configuration):
    set_seeds(cfg.seed)
    current_run_folder = Path(HydraConfig.get().runtime.output_dir)
    writer = SummaryWriter(log_dir=str(current_run_folder/"tb"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = setup_data(copy_to_current_dir=False)
    train_loader, valid_loader, idx_to_class, class_weights = create_dataloaders(
        data_path,
        pin_memory=torch.cuda.is_available(),
        cfg=cfg.data
    )
    n_classes = cfg.data.n_classes
    model = get_resnet_model(cfg=cfg.model, n_classes=n_classes, device=device)
    loss_fn = get_loss_fn(cfg.training.loss_fn, class_weights, n_classes, device)
    run_complete_training(
        cfg, train_loader, valid_loader, model, loss_fn, device, n_classes, idx_to_class, current_run_folder, writer
    )
    writer.close()

if __name__ == '__main__':
    main()