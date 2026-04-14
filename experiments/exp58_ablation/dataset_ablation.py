from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_CANDIDATES = ["label", "target", "class", "mineral", "mineral_class", "y"]
HUMIDITY_CANDIDATES = ["humidity", "rh", "h", "humidity_label"]
GROUP_CANDIDATES = ["source_file", "source_id", "source", "parent_id", "file_id", "group"]

def _find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def load_train_test(dataset_dir: str):
    dataset_dir = Path(dataset_dir)
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected train.csv and test.csv under {dataset_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_col = _find_col(list(train_df.columns), LABEL_CANDIDATES)
    humidity_col = _find_col(list(train_df.columns), HUMIDITY_CANDIDATES)
    group_col = _find_col(list(train_df.columns), GROUP_CANDIDATES)

    if label_col is None:
        raise ValueError("Could not detect label column.")
    if humidity_col is None:
        raise ValueError("Could not detect humidity column.")

    exclude = {label_col, humidity_col}
    if group_col is not None:
        exclude.add(group_col)

    feature_cols = []
    for col in train_df.columns:
        if col in exclude or col not in test_df.columns:
            continue
        tr = pd.to_numeric(train_df[col], errors="coerce")
        te = pd.to_numeric(test_df[col], errors="coerce")
        if tr.notna().all() and te.notna().all():
            feature_cols.append(col)

    labels = sorted(train_df[label_col].astype(str).unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    meta = {
        "label_col": label_col,
        "humidity_col": humidity_col,
        "group_col": group_col,
        "feature_cols": feature_cols,
        "label_to_idx": label_to_idx,
    }
    return train_df, test_df, meta

class NIRCSVDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, humidity_col, label_to_idx):
        self.X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).copy()
        self.h = pd.to_numeric(df[humidity_col], errors="coerce").to_numpy(dtype=np.float32).copy()
        labels = df[label_col].astype(str).tolist()
        self.y = np.array([label_to_idx[x] for x in labels], dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "spectrum": torch.tensor(self.X[idx], dtype=torch.float32),
            "humidity": torch.tensor(self.h[idx], dtype=torch.float32),
            "label": torch.tensor(self.y[idx], dtype=torch.long),
        }
