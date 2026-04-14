from pathlib import Path
import sys
import os
import json
import random
import argparse
from typing import Dict, List

CUR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CUR_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Models.HDA_CNN import HDA_CNN
from Models.CNN_1D import CNN_1D
from Models.Concat_CNN import Concat_CNN
from Models.ResNet1D import ResNet1D
from Utils.dataset_loader import SpectralDataset
from Utils.metrics import evaluate_metrics

DEFAULT_SEEDS = [42, 52, 62, 72, 82]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_type: str = "hda", num_classes: int = 4):
    if model_type == "hda":
        return HDA_CNN(num_classes=num_classes)
    if model_type == "cnn":
       return CNN_1D(num_classes=num_classes)
    if model_type == "concat":
        return Concat_CNN(num_classes=num_classes)
    if model_type == "resnet":
        return ResNet1D(num_classes=num_classes)
    raise ValueError(f"Unsupported model_type: {model_type}")


def infer_num_classes(dataset):
    if hasattr(dataset, "labels"):
        return len(set(int(x) for x in dataset.labels))
    if hasattr(dataset, "y"):
        return len(set(int(x) for x in dataset.y))
    label_set = set()
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        if torch.is_tensor(label):
            label = label.item()
        label_set.add(int(label))
    return len(label_set)


def compute_class_weights(dataset, num_classes: int):
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        if torch.is_tensor(label):
            label = label.item()
        counts[int(label)] += 1
    counts = np.maximum(counts, 1)
    weights = len(dataset) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def forward_by_model(model, model_type: str, spec, hum):
    if model_type in ["concat", "hda"]:
        return model(spec, hum)
    return model(spec)


@torch.no_grad()
def evaluate_model(model, loader, device, model_type: str = "hda"):
    model.eval()
    y_true, y_pred = [], []
    for spec, hum, label in loader:
        spec = spec.to(device)
        hum = hum.to(device)
        label = label.to(device)
        output = forward_by_model(model, model_type, spec, hum)
        pred = torch.argmax(output, dim=1)
        y_true.extend(label.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    acc, report, cm = evaluate_metrics(y_true, y_pred)
    macro_precision = float(report["macro avg"]["precision"])
    macro_recall = float(report["macro avg"]["recall"])
    macro_f1 = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])
    return {
        "accuracy": float(acc),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "cm": cm,
    }


def train_one_seed(model_type: str, dataset_dir: str, exp_name: str, seed: int, epochs: int = 60):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(dataset_dir, "train.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"test.csv not found: {test_csv}")

    train_set = SpectralDataset(train_csv)
    test_set = SpectralDataset(test_csv)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    num_classes = infer_num_classes(train_set)
    model = build_model(model_type=model_type, num_classes=num_classes).to(device)
    class_weights = compute_class_weights(train_set, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {"train_loss": []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for spec, hum, label in train_loader:
            spec = spec.to(device)
            hum = hum.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = forward_by_model(model, model_type, spec, hum)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            batch_size = label.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        scheduler.step()
        avg_loss = running_loss / max(total_samples, 1)
        history["train_loss"].append(float(avg_loss))
        print(
            f"[{exp_name}] [{model_type}] [seed={seed}] Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"Train Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    metrics = evaluate_model(model, test_loader, device, model_type=model_type)

    save_dir = Path("checkpoints") / exp_name / model_type / f"seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "final.pth")

    result_dir = Path("results") / exp_name / model_type / f"seed{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(result_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def resolve_split_dir(dataset_root: Path, split_value: str) -> Path:
    p = Path(str(split_value))
    if p.exists():
        return p
    alt = dataset_root / p.name
    if alt.exists():
        return alt
    alt2 = dataset_root / str(split_value)
    if alt2.exists():
        return alt2
    raise FileNotFoundError(f"Cannot resolve split dir: {split_value}")


def aggregate_split_seed_metrics(rows: List[Dict]) -> Dict[str, float]:
    df = pd.DataFrame(rows)
    out = {}
    for col in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
        out[f"{col}_mean"] = float(df[col].mean())
        out[f"{col}_sd"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def summarize_over_splits(split_mean_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_bases = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    for model, g in split_mean_df.groupby("model"):
        row = {"model": model, "n_splits": int(len(g))}
        for base in metric_bases:
            row[f"{base}_mean_over_splits"] = float(g[f"{base}_mean"].mean())
            row[f"{base}_sd_over_splits"] = float(g[f"{base}_mean"].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1_mean_over_splits", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Path to repeated holdout root containing manifest.csv")
    parser.add_argument("--output-dir", required=True, help="Directory to save aggregated csv outputs")
    parser.add_argument("--models", default="hda,resnet,concat,cnn", help="Comma-separated: hda,resnet,concat,cnn")
    parser.add_argument("--seeds", default="42,52,62,72,82", help="Comma-separated seed list per split")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--exp-prefix", default="repeated_holdout_5seeds")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    raw_rows = []
    split_mean_rows = []

    for _, rec in manifest.iterrows():
        split_seed = int(rec["split_seed"]) if "split_seed" in rec else None
        split_dir = resolve_split_dir(dataset_root, rec["split_dir"])
        split_name = split_dir.name
        print(f"\n===== Running split: {split_name} =====")

        for model_type in model_list:
            seed_metrics = []
            for seed in seeds:
                exp_name = f"{args.exp_prefix}/{split_name}"
                metrics = train_one_seed(
                    model_type=model_type,
                    dataset_dir=str(split_dir),
                    exp_name=exp_name,
                    seed=seed,
                    epochs=args.epochs,
                )
                row = {
                    "split_name": split_name,
                    "split_seed": split_seed,
                    "model": model_type,
                    "seed": seed,
                    **metrics,
                }
                raw_rows.append(row)
                seed_metrics.append(row)

            split_stats = aggregate_split_seed_metrics(seed_metrics)
            split_mean_rows.append(
                {
                    "split_name": split_name,
                    "split_seed": split_seed,
                    "model": model_type,
                    "n_seeds": len(seeds),
                    **split_stats,
                }
            )
            print(
                f"[{split_name}] [{model_type}] split-mean: "
                f"acc={split_stats['accuracy_mean']:.4f}±{split_stats['accuracy_sd']:.4f}, "
                f"macro_f1={split_stats['macro_f1_mean']:.4f}±{split_stats['macro_f1_sd']:.4f}"
            )

    raw_df = pd.DataFrame(raw_rows)
    split_mean_df = pd.DataFrame(split_mean_rows)
    summary_df = summarize_over_splits(split_mean_df)

    raw_df.to_csv(output_dir / "seed_metrics.csv", index=False)
    split_mean_df.to_csv(output_dir / "split_mean_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary_over_splits.csv", index=False)

    print("\nSaved:")
    print(output_dir / "seed_metrics.csv")
    print(output_dir / "split_mean_metrics.csv")
    print(output_dir / "summary_over_splits.csv")


if __name__ == "__main__":
    main()
