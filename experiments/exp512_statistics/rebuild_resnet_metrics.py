from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# ===== 关键：把项目根目录加入 sys.path =====
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.ResNet1D import ResNet1D


LABEL_CANDIDATES = ["label", "target", "class", "mineral", "mineral_class", "y"]
HUMIDITY_CANDIDATES = ["humidity", "rh", "h", "humidity_label"]
GROUP_CANDIDATES = ["source_file", "source_id", "source", "parent_id", "file_id", "group"]


def find_col(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_test_csv(dataset_dir: str):
    test_path = Path(dataset_dir) / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"找不到测试集文件: {test_path}")

    test_df = pd.read_csv(test_path)

    label_col = find_col(test_df.columns, LABEL_CANDIDATES)
    humidity_col = find_col(test_df.columns, HUMIDITY_CANDIDATES)
    group_col = find_col(test_df.columns, GROUP_CANDIDATES)

    if label_col is None:
        raise ValueError("未识别到 label 列")
    if humidity_col is None:
        raise ValueError("未识别到 humidity 列")

    exclude = {label_col, humidity_col}
    if group_col is not None:
        exclude.add(group_col)

    feature_cols = []
    for col in test_df.columns:
        if col in exclude:
            continue
        s = pd.to_numeric(test_df[col], errors="coerce")
        if s.notna().all():
            feature_cols.append(col)

    labels = sorted(test_df[label_col].astype(str).unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    return test_df, feature_cols, label_col, label_to_idx, idx_to_label


class TestDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, label_to_idx):
        self.X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.y = np.array([label_to_idx[x] for x in df[label_col].astype(str)], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def build_model(num_classes: int):
    # 如需改初始化参数，可在这里改
    model = ResNet1D(num_classes=num_classes)
    return model


def load_checkpoint(model, ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    return model


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).unsqueeze(1)  # [B,1,L]
            logits = model(x)
            pred = logits.argmax(dim=1)

            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    return y_true, y_pred


def parse_seed_from_filename(path: Path):
    m = re.search(r"seed[_\-]?(\d+)", path.stem.lower())
    if m:
        return int(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="Dataset/NIR_cross_humidity")
    parser.add_argument("--output_root", type=str, default="results/nir_crosshum_v2/resnet")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint 文件: {ckpt_path}")

    seed = parse_seed_from_filename(ckpt_path)
    if seed is None:
        raise ValueError(f"无法从文件名中解析 seed: {ckpt_path.name}")

    out_dir = Path(args.output_root) / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df, feature_cols, label_col, label_to_idx, idx_to_label = load_test_csv(args.dataset_dir)
    test_set = TestDataset(test_df, feature_cols, label_col, label_to_idx)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(label_to_idx)).to(device)
    model = load_checkpoint(model, ckpt_path, device)

    y_true, y_pred = evaluate(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "seed": seed,
        "model": "resnet",
        "dataset_dir": args.dataset_dir,
        "test_accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_label": [idx_to_label[i] for i in y_true],
        "y_pred_label": [idx_to_label[i] for i in y_pred],
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()