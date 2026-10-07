from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("training/checkpoints"))
    return p.parse_args()


def build_model():
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model, weights.transforms()


def main():
    cfg = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, eval_transform = build_model()
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=eval_transform.mean, std=eval_transform.std),
    ])

    train_ds = datasets.ImageFolder(cfg.data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(cfg.data_dir / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(cfg.data_dir / "test", transform=eval_transform)

    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise RuntimeError("train/val/test class folders do not match")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    def evaluate(loader):
        model.eval()
        ys, preds, probs = [], [], []
        total_loss = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                total_loss += criterion(logits, y).item() * y.size(0)
                p = torch.softmax(logits, dim=1)[:, 1]
                ys.extend(y.cpu().numpy().tolist())
                preds.extend(logits.argmax(1).cpu().numpy().tolist())
                probs.extend(p.cpu().numpy().tolist())
        metrics = {
            "loss": total_loss / max(1, len(loader.dataset)),
            "accuracy": accuracy_score(ys, preds),
            "precision": precision_score(ys, preds, zero_division=0),
            "recall": recall_score(ys, preds, zero_division=0),
            "f1": f1_score(ys, preds, zero_division=0),
            "roc_auc": roc_auc_score(ys, probs) if len(set(ys)) == 2 else None,
        }
        return metrics, np.array(ys), np.array(preds), np.array(probs)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * y.size(0)

        val_metrics, _, _, _ = evaluate(val_loader)
        row = {"epoch": epoch, "train_loss": running / max(1, len(train_loader.dataset)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, cfg.output_dir / "unmaskd_video_best.pt")

    checkpoint = torch.load(cfg.output_dir / "unmaskd_video_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, y_true, y_pred, y_prob = evaluate(test_loader)
    print("TEST", test_metrics)

    with open(cfg.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "test": test_metrics, "class_to_idx": train_ds.class_to_idx}, f, indent=2)

    report = classification_report(y_true, y_pred, target_names=[k for k, _ in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])], digits=4)
    (cfg.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Unmaskd video detector confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(cfg.output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
