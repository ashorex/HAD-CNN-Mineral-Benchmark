import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from exp510_dataset import generate_split_tables, GeneratedSplitDataset, PRESET_SPLITS
from Models.HDA_CNN import HDA_CNN

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
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
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

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 5.10 alternative non_overlapping humidity splits")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--preset", type=str, required=True, choices=list(PRESET_SPLITS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--results_root", type=str, default="results/exp510_non_overlapping")
    parser.add_argument("--n_train_aug", type=int, default=10)
    parser.add_argument("--n_test_repeat", type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    generated_train, generated_test, meta = generate_split_tables(
        dataset_dir=args.dataset_dir,
        preset_name=args.preset,
        n_train_aug=args.n_train_aug,
        n_test_repeat=args.n_test_repeat,
        random_seed=args.seed,
    )

    feature_cols = meta["feature_cols"]
    label_col = meta["label_col"]

    train_set = GeneratedSplitDataset(generated_train, feature_cols, label_col, "humidity", meta["label_to_idx"])
    test_set = GeneratedSplitDataset(generated_test, feature_cols, label_col, "humidity", meta["label_to_idx"])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    num_classes = len(meta["label_to_idx"])
    model = HDA_CNN(num_classes=num_classes).to(device)

    class_weights = compute_class_weights(train_set.y, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.results_root) / args.preset / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Experiment 5.10 alternative non_overlapping humidity splits")
    print(f"Preset        : {args.preset}")
    print(f"Description   : {meta['preset_description']}")
    print(f"Train levels  : {meta['train_levels']}")
    print(f"Test levels   : {meta['test_levels']}")
    print(f"Seed          : {args.seed}")
    print(f"Dataset dir   : {args.dataset_dir}")
    print(f"Train shape   : {generated_train.shape}")
    print(f"Test shape    : {generated_test.shape}")
    print(f"Device        : {device}")
    print("=" * 80)

    log_rows = []
    for epoch in range(1, args.epochs + 1):
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

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[{args.preset}][seed={args.seed}] epoch={epoch:03d} loss={train_loss:.4f} acc={test_metrics['accuracy']:.4f} f1={test_metrics['macro_f1']:.4f}")

    final_metrics, y_true, y_pred = evaluate(model, test_loader, device)

    (out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)
    generated_train.to_csv(out_dir / "generated_train.csv", index=False)
    generated_test.to_csv(out_dir / "generated_test.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("Finished.")
    print(final_metrics)
    print(f"Saved to: {out_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
