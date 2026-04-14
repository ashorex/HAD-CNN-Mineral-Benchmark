from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_CANDIDATES = ["label", "target", "class", "mineral", "mineral_class", "y"]
HUMIDITY_CANDIDATES = ["humidity", "rh", "h", "humidity_label"]
GROUP_CANDIDATES = ["source_file", "source_id", "source", "parent_id", "file_id", "group"]

PRESET_SPLITS = {
    "split_A_current": {
        "train_levels": [0.1, 0.3, 0.5, 0.7, 0.9],
        "test_levels":  [0.0, 0.2, 0.4, 0.6, 0.8],
        "description": "Current odd-train/even-test non_overlapping split"
    },
    "split_B_shifted": {
        "train_levels": [0.00, 0.18, 0.36, 0.54, 0.72],
        "test_levels":  [0.09, 0.27, 0.45, 0.63, 0.81],
        "description": "Alternative staggered non-overlapping split within the same humidity range"
    },
    "split_C_shifted": {
        "train_levels": [0.05, 0.23, 0.41, 0.59, 0.77],
        "test_levels":  [0.14, 0.32, 0.50, 0.68, 0.86],
        "description": "Second alternative staggered non_overlapping split"
    },
}

DEFAULT_PERTURB = {
    "alpha": 2.0,
    "delta": 0.0015,
    "sigma0": 0.8,
    "eta": 1.2,
    "kappa": 0.02,
}

def _find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def load_base_train_test(dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
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
    if group_col is None:
        raise ValueError("Could not detect source/group column. A source-file-level split is required for Experiment 5.10.")

    exclude = {label_col, humidity_col, group_col}
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
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    meta = {
        "label_col": label_col,
        "humidity_col": humidity_col,
        "group_col": group_col,
        "feature_cols": feature_cols,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
    }
    return train_df, test_df, meta

def wavelength_grid(n_features: int, wl_min: float = 0.35, wl_max: float = 2.5) -> np.ndarray:
    return np.linspace(wl_min, wl_max, n_features, dtype=np.float32)

def water_response(wl: np.ndarray, peak_width: float = 0.015) -> np.ndarray:
    centers = [1.40, 1.90]
    resp = np.zeros_like(wl, dtype=np.float32)
    for c in centers:
        resp += np.exp(-0.5 * ((wl - c) / peak_width) ** 2).astype(np.float32)
    resp /= (resp.max() + 1e-8)
    return resp

def gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    radius = max(int(truncate * sigma + 0.5), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2)).astype(np.float32)
    kernel /= (kernel.sum() + 1e-8)
    return kernel

def gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel_1d(sigma)
    pad = len(kernel) // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    y = np.convolve(x_pad, kernel, mode="valid")
    return y.astype(np.float32)

def apply_humidity_perturbation(
    spectrum: np.ndarray,
    humidity: float,
    wl: np.ndarray,
    alpha: float,
    delta: float,
    sigma0: float,
    eta: float,
    kappa: float,
) -> np.ndarray:
    x = spectrum.astype(np.float32).copy()
    h = float(humidity)
    W = water_response(wl)

    x = x * np.exp(-alpha * h * W).astype(np.float32)
    shift = delta * h
    x = np.interp(wl - shift, wl, x, left=float(x[0]), right=float(x[-1])).astype(np.float32)
    sigma = sigma0 + eta * h
    x = gaussian_smooth_1d(x, sigma=sigma)
    x = x + (kappa * h * (wl - wl.min())).astype(np.float32)
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)

def random_augment_training(x: np.ndarray, wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = x.astype(np.float32).copy()
    s = rng.uniform(0.98, 1.02)
    out = out * np.float32(s)

    eps = rng.uniform(-0.001, 0.001)
    out = np.interp(wl - eps, wl, out, left=float(out[0]), right=float(out[-1])).astype(np.float32)

    a = rng.uniform(-0.01, 0.01)
    b = rng.uniform(-0.01, 0.01)
    baseline = a * (wl - wl.min()) + b
    out = out + baseline.astype(np.float32)

    sigma = rng.uniform(0.001, 0.005)
    noise = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
    out = out + noise

    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)

def build_source_templates(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    group_col: str,
) -> pd.DataFrame:
    feature_mean = df.groupby(group_col, as_index=False)[feature_cols].mean()
    label_mode = df.groupby(group_col)[label_col].agg(lambda x: x.astype(str).mode().iat[0]).reset_index()
    merged = feature_mean.merge(label_mode, on=group_col, how="left")
    return merged

def generate_split_tables(
    dataset_dir: str,
    preset_name: str,
    n_train_aug: int = 10,
    n_test_repeat: int = 10,
    random_seed: int = 42,
):
    train_df, test_df, meta = load_base_train_test(dataset_dir)

    if preset_name not in PRESET_SPLITS:
        raise ValueError(f"Unsupported preset_name: {preset_name}")

    preset = PRESET_SPLITS[preset_name]
    train_levels = preset["train_levels"]
    test_levels = preset["test_levels"]

    feature_cols = meta["feature_cols"]
    label_col = meta["label_col"]
    group_col = meta["group_col"]

    train_templates = build_source_templates(train_df, feature_cols, label_col, group_col)
    test_templates = build_source_templates(test_df, feature_cols, label_col, group_col)

    wl = wavelength_grid(len(feature_cols))
    rng = np.random.default_rng(random_seed)

    def _generate(df_templates: pd.DataFrame, levels: List[float], is_train: bool):
        rows = []
        repeat_n = n_train_aug if is_train else n_test_repeat

        for _, row in df_templates.iterrows():
            base_spec = row[feature_cols].to_numpy(dtype=np.float32)
            label = row[label_col]
            source_file = row[group_col]

            for h in levels:
                deterministic = apply_humidity_perturbation(
                    spectrum=base_spec,
                    humidity=h,
                    wl=wl,
                    alpha=DEFAULT_PERTURB["alpha"],
                    delta=DEFAULT_PERTURB["delta"],
                    sigma0=DEFAULT_PERTURB["sigma0"],
                    eta=DEFAULT_PERTURB["eta"],
                    kappa=DEFAULT_PERTURB["kappa"],
                )

                for rep in range(repeat_n):
                    spec = random_augment_training(deterministic, wl, rng) if is_train else deterministic.copy()
                    item = {
                        label_col: label,
                        "humidity": float(h),
                        group_col: source_file,
                        "repeat_id": rep,
                    }
                    for c, v in zip(feature_cols, spec):
                        item[c] = float(v)
                    rows.append(item)

        return pd.DataFrame(rows)

    new_train = _generate(train_templates, train_levels, is_train=True)
    new_test = _generate(test_templates, test_levels, is_train=False)

    new_meta = dict(meta)
    new_meta["preset_name"] = preset_name
    new_meta["preset_description"] = preset["description"]
    new_meta["train_levels"] = train_levels
    new_meta["test_levels"] = test_levels
    return new_train, new_test, new_meta

class GeneratedSplitDataset(Dataset):
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
