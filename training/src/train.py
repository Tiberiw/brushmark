import torch
import torch.nn as nn
from data_setup import create_dataloaders
from download_data import setup_data
from model import get_resnet50_model
from engine import train
from pathlib import Path
from datetime import datetime
import os

STAGE_ONE_EPOCHS = 1
STAGE_TWO_EPOCHS = 25
LABEL_SMOOTHING = 0.1
BATCH_SIZE = 32
CPU_COUNT = os.cpu_count() or 1
NUM_WORKERS = 4 if CPU_COUNT > 4 else CPU_COUNT

if __name__ == '__main__':
    runs_dir = Path("training")/"runs"
    runs_dir.mkdir(exist_ok=True)

    current_run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Preparing folder for run: {current_run}")
    current_run_folder = runs_dir/f"{current_run}"
    current_run_folder.mkdir(exist_ok=True)

    data_path = setup_data(copy_to_current_dir=False)
    train_loader, valid_loader, idx_to_class, class_weights = create_dataloaders(
        data_path, bs=BATCH_SIZE, num_workers=NUM_WORKERS)
    n_classes = len(set(idx_to_class.values()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet50_model(n_classes, device)

    # penalize minority errors more, but not as extreme as linear (sqrt dampening)
    weights = torch.tensor(
        [class_weights[i] ** 0.5 for i in range(n_classes)],
        dtype=torch.float,
        device=device
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        params=model.fc.parameters(), lr=2e-3, weight_decay=0.01
    )
    loss_fn_sample = nn.CrossEntropyLoss(reduction='none') # Used for extracting top losses per sample

    train(
        STAGE_ONE_EPOCHS,
        train_loader,
        valid_loader,
        model,
        optimizer,
        loss_fn,
        loss_fn_sample,
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
        optimizer_full, T_max=STAGE_TWO_EPOCHS, eta_min=1e-6
    )
    save = {
        'mx': 0.75,
        'path': ''
    }

    train(
        STAGE_TWO_EPOCHS,
        train_loader,
        valid_loader,
        model,
        optimizer_full,
        loss_fn,
        loss_fn_sample,
        device,
        n_classes,
        idx_to_class,
        current_run_folder,
        scheduler,
        save
        )