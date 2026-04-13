import matplotlib.pyplot as plt
from utils import ema_smoothing
import torch
from pathlib import Path

def get_metrics_from_cm(cm: torch.Tensor) -> list[tuple[float, float, float]]:
    """ Function for extracting the Precision, Recall and F1 score from confusion matrix """
    metrics = []
    eps = 1e-8
    for i in range(cm.shape[0]):
        TP = cm[i, i].item()
        FP = cm[:, i].sum().item() - TP
        FN = cm[i].sum().item() - TP
        Precision = TP / (TP + FP + eps)
        Recall = TP / (TP + FN + eps)
        F1 = 2 * Precision * Recall / (Precision + Recall + eps)
        metrics.append([F1, Precision, Recall])
    return metrics

def log_per_class_metrics(cm: torch.Tensor, idx_to_class: dict[int, str]) -> None:
    """ Function to display metrics per class, sorted by F1, using confusion matrix"""
    metrics = get_metrics_from_cm(cm)
    order = torch.argsort(torch.tensor(metrics)[:, 0]).tolist()
    for idx in order:
        F1, Precision, Recall = metrics[idx]
        print(f"{idx_to_class[idx]}: F1={F1:.4f}, Precision={Precision:.4f}, Recall={Recall:.4f}")

def plot_top_confusions(cm: torch.Tensor,
                        idx_to_class: dict[int, str],
                        save_path: Path,
                        k: int=10) -> None:
    pairs = {}
    ln = cm.shape[0]
    for i in range(ln - 1):
        for j in range(i + 1, ln):
            pairs[(i, j)] = cm[i, j].item() + cm[j, i].item()
    pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:k]

    labels = [f"{idx_to_class[i]} ↔ {idx_to_class[j]}" for (i, j), _ in pairs]
    counts = [c for _, c in pairs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], counts[::-1])
    ax.set_xlabel('Misclassifications')
    ax.set_title(f'Top {k} Confused Class Pairs')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_top_losses(top_losses: dict, save_path: Path) -> None:
    """ Function to plot top losses"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    preds = top_losses['preds']
    actuals = top_losses['actuals']
    losses = top_losses['losses']
    images = top_losses['images']
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    
    for idx, ax in enumerate(axes.flat):
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        img = img * std.numpy() + mean.numpy()
        img = img.clip(0, 1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(
            f"Pred: {preds[idx]} | Actual: {actuals[idx]} | Loss: {losses[idx]:.4f}",
            fontsize=10
        )
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_performance(train_losses: list[float],
                     valid_losses: list[float],
                     F1s: list[float],
                     accs: list[float],
                     save_path: Path) -> None:
    """Function for plotting the performance graphs"""
    train_losses_ema = ema_smoothing(train_losses)
    train_idx = list(range(len(train_losses_ema)))
    valid_idx = list(range(len(valid_losses)))
    
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(train_idx, train_losses, alpha=0.3, label='raw')
    ax[0, 0].plot(train_idx, train_losses_ema, color='red', label='EMA')
    ax[0, 0].set_title('Train loss')
    ax[0, 1].plot(valid_idx, valid_losses)
    ax[0, 1].set_title('Valid loss')
    ax[1, 0].plot(valid_idx, F1s)
    ax[1, 0].set_title('Valid F1 scores')
    ax[1, 1].plot(valid_idx, accs)
    ax[1, 1].set_title('Valid Accuracy')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)