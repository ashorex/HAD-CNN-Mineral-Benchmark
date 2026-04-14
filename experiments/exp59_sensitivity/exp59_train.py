import argparse
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

from Sensitivity.exp59_dataset import load_train_test_csvs, SensitivityDataset

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
    parser = argparse.ArgumentParser(description="Experiment 5.9 sensitivity analysis")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--param", type=str, required=True, choices=["alpha", "delta", "sigma0", "eta", "kappa"])
    parser.add_argument("--value", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--results_root", type=str, default="results/exp59_sensitivity")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    train_df, test_df, meta = load_train_test_csvs(args.dataset_dir)

    train_set = SensitivityDataset(
        train_df, meta["feature_cols"], meta["label_col"], meta["humidity_col"], meta["label_to_idx"],
        param_name=args.param, param_value=args.value
    )
    test_set = SensitivityDataset(
        test_df, meta["feature_cols"], meta["label_col"], meta["humidity_col"], meta["label_to_idx"],
        param_name=args.param, param_value=args.value
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    num_classes = len(meta["label_to_idx"])
    model = HDA_CNN(num_classes=num_classes).to(device)

    class_weights = compute_class_weights(train_set.y, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    value_tag = f"{args.value:.6f}".rstrip("0").rstrip(".")
    out_dir = Path(args.results_root) / args.param / f"value_{value_tag}" / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Experiment 5.9 sensitivity analysis")
    print(f"Parameter     : {args.param}")
    print(f"Value         : {args.value}")
    print(f"Seed          : {args.seed}")
    print(f"Dataset dir   : {args.dataset_dir}")
    print(f"Feature dims  : {len(meta['feature_cols'])}")
    print(f"Train shape   : {train_set.X.shape}")
    print(f"Test shape    : {test_set.X.shape}")
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
            print(
                f"[{args.param}={args.value}][seed={args.seed}] "
                f"epoch={epoch:03d} "
                f"loss={train_loss:.4f} "
                f"acc={test_metrics['accuracy']:.4f} "
                f"f1={test_metrics['macro_f1']:.4f}"
            )

    final_metrics, y_true, y_pred = evaluate(model, test_loader, device)

    (out_dir / "metrics.json").write_text(
        json.dumps(final_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("=" * 80)
    print("Finished.")
    print(final_metrics)
    print(f"Saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
