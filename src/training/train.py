"""Train the fake ID classifier."""

import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np

from src.data.dataset import IDDataset, get_train_transforms, get_eval_transforms
from src.models.classifier import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def evaluate(model, loader, device, criterion):
    model.eval()
    losses, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = logits.argmax(axis=1)
    return {
        "loss": float(np.mean(losses)),
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
    }


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    for k in ("lr_head", "lr_backbone", "weight_decay"):
        cfg["training"][k] = float(cfg["training"][k])

    torch.manual_seed(cfg["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    use_amp = cfg["training"]["use_amp"] and device.type == "cuda"

    # Data
    train_ds = IDDataset(resolve(cfg["data"]["train_csv"]),
                         transforms=get_train_transforms(cfg["data"]["image_size"]))
    val_ds = IDDataset(resolve(cfg["data"]["val_csv"]),
                       transforms=get_eval_transforms(cfg["data"]["image_size"]))

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    # Model
    model = build_model(num_classes=cfg["model"]["num_classes"],
                        freeze_backbone=cfg["model"]["freeze_backbone"],
                        dropout=cfg["model"]["dropout"]).to(device)

    # Class weights for imbalance
    counts = train_ds.df["label"].value_counts().sort_index().values
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Class weights: {weights.cpu().numpy()}")

    # Two LR groups: head higher, backbone lower
    head_params = [p for n, p in model.named_parameters() if n.startswith("classifier")]
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    optimizer = AdamW([
        {"params": head_params, "lr": cfg["training"]["lr_head"]},
        {"params": backbone_params, "lr": cfg["training"]["lr_backbone"]},
    ], weight_decay=cfg["training"]["weight_decay"])

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = GradScaler("cuda") if use_amp else None

    out_dir = resolve(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with autocast("cuda"):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"Epoch {epoch+1:02d} | train loss {train_loss:.4f} | "
              f"val loss {val_metrics['loss']:.4f} | acc {val_metrics['acc']:.4f} | "
              f"f1 {val_metrics['f1']:.4f} | auc {val_metrics['auc']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": val_metrics, "config": cfg},
                       out_dir / "best.pt")
            print(f"  -> saved best to {out_dir / 'best.pt'}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args.config)