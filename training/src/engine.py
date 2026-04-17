from tqdm import tqdm
import torch
import torch.nn as nn
from metrics import get_metrics_from_cm, log_per_class_metrics, plot_performance, plot_top_confusions, plot_top_losses
from pathlib import Path

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> list[float]:
    """Function for training the model for one epoch on device. The data is sampled fomr the dataloader.
    The optimization is done with respect to the loss_fn and the weights are updated by the optimizer.
        Returns: list[float] - losses per batch
    """
    train_losses = []
    model.train()
    for batch, targets in tqdm(dataloader):
        batch = batch.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        preds = model(batch)
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
    return train_losses

def validate_step(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device: torch.device,
                  n_classes: int,
                  idx_to_class: dict[int, str]) -> tuple[float, torch.Tensor, dict]:
    """Function for validating the model the validation set after one epoch of training.
        Returns: tuple[float, torch.tensor, dict] - returns performance metrics on the validation set
        such as the everage error for the validation set, the confusion matrix and top losses
    """
    val_losses = []
    cm = torch.zeros((n_classes, n_classes), dtype=torch.long, device=device)
    model.eval()
    all_metrics = []
    all_images = []
    loss_fn_sample = nn.CrossEntropyLoss(reduction='none') # Used for extracting top losses per sample
    
    with torch.no_grad():
        for batch, targets in tqdm(dataloader):
            batch = batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(batch)
            loss = loss_fn(outputs, targets)
            val_losses.append(loss.item())
            predictions = torch.softmax(outputs, dim=-1).argmax(dim=-1)

            # Compute confusion matrix
            indices = targets * n_classes + predictions
            cm += torch.bincount(indices, minlength=n_classes ** 2).reshape(n_classes, n_classes)
            
            # Compute top k losses
            loss_samples = loss_fn_sample(outputs, targets)
            k = 6
            top_k_idx = torch.argsort(loss_samples, descending=True)[:k]
            for i in top_k_idx:
                all_metrics.append((predictions[i].item(), targets[i].item(), loss_samples[i].item()))
                all_images.append(batch[i].cpu())

        # global top-k
        all_metrics_sorted = sorted(enumerate(all_metrics), key=lambda x: x[1][2], reverse=True)[:6]
        top_indices = [i for i, _ in all_metrics_sorted]
        top_losses = {
            'preds':   [idx_to_class[all_metrics[i][0]] for i in top_indices],
            'actuals': [idx_to_class[all_metrics[i][1]] for i in top_indices],
            'losses':  [all_metrics[i][2] for i in top_indices],
            'images':  torch.stack([all_images[i] for i in top_indices]),
        }
        val_loss = sum(val_losses) / len(val_losses)
    return val_loss, cm, top_losses

def train(epochs: int,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          device: torch.device,
          n_classes: int,
          idx_to_class: dict[int, str],
          current_run_folder: Path,
          scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
          save: dict | None = None) -> None:
    """Main training loop function. Trains the model for epoch epochs. For each epoch. Calls the trainig
    and validation functions. At the end display results
    """
    all_train_losses, all_valid_losses, F1s, accs = [], [], [], []
    for epoch in range(epochs):
        train_losses = train_step(model, train_dataloader, loss_fn, optimizer, device)
        valid_loss, cm, top_losses = validate_step(
            model, valid_dataloader, loss_fn, device, n_classes, idx_to_class
        )
        if scheduler: scheduler.step()
        
        f1s = torch.tensor([F1 for (F1, _, _) in get_metrics_from_cm(cm)]).to(device, non_blocking=True)
        counts_per_class = cm.sum(dim=1)
        macro_avg_F1 = torch.mean(f1s).item()
        weighted_F1 = torch.sum(f1s * counts_per_class).item() / cm.sum().item()
        train_loss = torch.mean(torch.tensor(train_losses)).item()
        valid_acc = (cm.diag().sum() / cm.sum()).item()
        print(f"Epoch {epoch} - Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f} | Valid Accuracy: {valid_acc:.4f} | Macro-Avg F1: {macro_avg_F1:.4f} | Weighted F1: {weighted_F1:.4f}")
        if scheduler:
            print(F"Learning rate scheduler: {scheduler.get_last_lr()}")

        if save and macro_avg_F1 > save['mx']:
            save['mx'] = macro_avg_F1
            save_path = current_run_folder / "best_model.pt"
            save['path'] = save_path
            torch.save(model.state_dict(), save_path)
            print(f"Saved model on epoch: {epoch}, with F1: {macro_avg_F1}, at: {save_path}")
        
        if epoch == epochs - 1:
            log_per_class_metrics(cm, idx_to_class)
            plot_top_confusions(cm, idx_to_class, current_run_folder/"top_confusions.png", 10)
            plot_top_losses(top_losses, current_run_folder/"top_losses.png")
    
        all_train_losses.extend(train_losses)
        all_valid_losses.append(valid_loss)
        F1s.append(macro_avg_F1)
        accs.append(valid_acc)
    plot_performance(all_train_losses, all_valid_losses, F1s, accs, current_run_folder/"performance.png")