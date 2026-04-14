import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from Ablation_study.dataset_ablation import load_train_test, NIRCSVDataset
from Ablation_study.HDA_CNN_ablation import HDA_CNN_Ablation

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

def macro_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x_spec = batch["spectrum"].to(device).unsqueeze(1)
            x_hum = batch["humidity"].to(device).unsqueeze(1)
            y = batch["label"].to(device)
            logits = model(x_spec, x_hum)
            pred = logits.argmax(dim=1)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())
    return macro_metrics(ys, ps), ys, ps

def train_nir_ablation(dataset_dir, exp_name, seed, epochs, variant, results_root="results/exp58_ablation", batch_size=32, lr=2e-4, weight_decay=1e-4, hum_dim=16):
    seed_everything(seed)

    train_df, test_df, meta = load_train_test(dataset_dir)
    train_set = NIRCSVDataset(train_df, meta["feature_cols"], meta["label_col"], meta["humidity_col"], meta["label_to_idx"])
    test_set = NIRCSVDataset(test_df, meta["feature_cols"], meta["label_col"], meta["humidity_col"], meta["label_to_idx"])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    num_classes = len(meta["label_to_idx"])
    model = HDA_CNN_Ablation(num_classes=num_classes, hum_dim=hum_dim, variant=variant).to(device)

    class_weights = compute_class_weights(train_set.y, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    out_dir = Path(results_root) / variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Variant       : {variant}")
    print(f"Seed          : {seed}")
    print(f"Dataset dir   : {dataset_dir}")
    print(f"Feature dims  : {len(meta['feature_cols'])}")
    print(f"Train shape   : {train_set.X.shape}")
    print(f"Test shape    : {test_set.X.shape}")
    print(f"Device        : {device}")
    print("=" * 80)

    log_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in train_loader:
            x_spec = batch["spectrum"].to(device).unsqueeze(1)
            x_hum = batch["humidity"].to(device).unsqueeze(1)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(x_spec, x_hum)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_n += bs

        train_loss = total_loss / max(total_n, 1)
        test_metrics, _, _ = evaluate(model, test_loader, device)
        log_rows.append({"epoch": epoch, "train_loss": train_loss, **test_metrics})

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"[{variant}][seed={seed}] epoch={epoch:03d} loss={train_loss:.4f} acc={test_metrics['accuracy']:.4f} f1={test_metrics['macro_f1']:.4f}")

    final_metrics, y_true, y_pred = evaluate(model, test_loader, device)

    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "variant": variant, "seed": seed}, out_dir / "model_final.pt")

    print("=" * 80)
    print("Finished.")
    print(final_metrics)
    print(f"Saved to: {out_dir}")
    print("=" * 80)
