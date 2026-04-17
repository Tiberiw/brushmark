import torch
import torch.nn as nn
from data_setup import create_dataloaders
from download_data import setup_data
from model import get_resnet50_model
from engine import train
from pathlib import Path
from datetime import datetime
import os
import argparse

cpu_count = os.cpu_count() or 1
NUM_WORKERS = 4 if cpu_count > 4 else cpu_count

def get_current_run_folder(runs_dir: Path) -> Path:
    """Function for creating the results folder for a training run.
        Return: Path to the results folder
    """

    current_run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Preparing folder for run: {current_run}")
    current_run_folder = runs_dir/f"{current_run}"
    current_run_folder.mkdir(exist_ok=True)
    return current_run_folder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_phase", action="store_true", help="Warmup phase")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.5, help="Head dropout")
    parser.add_argument("--epochs", type=int, default=25, required=True, help="Number of epochs")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label Smoothing")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    WARMUP_PHASE = args.warmup_phase
    BATCH_SIZE = args.batch_size
    DROPOUT = args.dropout
    EPOCHS = args.epochs
    LABEL_SMOOTHING = args.label_smoothing
    print(
        f"Training with:\n"
        f"Warmup phase: {WARMUP_PHASE}\n"
        f"Batch size: {BATCH_SIZE}\n"
        f"Dropout: {DROPOUT}\n"
        f"Epochs: {EPOCHS}\n"
        f"Label Smoothing: {LABEL_SMOOTHING}\n"
        f"==================================\n"
          )

    runs_dir = Path("training")/"runs"
    runs_dir.mkdir(exist_ok=True)
    current_run_folder = get_current_run_folder(runs_dir)

    data_path = setup_data(copy_to_current_dir=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, idx_to_class, class_weights = create_dataloaders(
        data_path,
        pin_memory=torch.cuda.is_available(),
        bs=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    n_classes = len(idx_to_class.values())
    model = get_resnet50_model(n_classes, device, dropout=DROPOUT)

    # penalize minority errors more, but not as extreme as linear (sqrt dampening)
    weights = torch.tensor(
        [class_weights[i] ** 0.5 for i in range(n_classes)],
        dtype=torch.float,
        device=device
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

    if WARMUP_PHASE:
        optimizer = torch.optim.AdamW(
            params=model.fc.parameters(), lr=2e-3, weight_decay=0.01
        )
        train(
            3,
            train_loader,
            valid_loader,
            model,
            optimizer,
            loss_fn,
            device,
            n_classes,
            idx_to_class,
            current_run_folder
            )

    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    optimizer_full = torch.optim.AdamW([
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_full, T_max=EPOCHS, eta_min=1e-6
    )
    save = {
        'mx': 0.75,
        'path': ''
    }

    train(
        EPOCHS,
        train_loader,
        valid_loader,
        model,
        optimizer_full,
        loss_fn,
        device,
        n_classes,
        idx_to_class,
        current_run_folder,
        scheduler,
        save
        )