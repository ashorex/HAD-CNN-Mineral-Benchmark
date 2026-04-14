from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LABEL_CANDIDATES = ["label", "target", "class", "mineral", "mineral_class", "y"]
HUMIDITY_CANDIDATES = ["humidity", "rh", "h", "humidity_label"]
GROUP_CANDIDATES = ["source_file", "source_id", "source", "parent_id", "file_id", "group"]

DEFAULT_PARAMS = {
    "alpha": 2.0,
    "delta": 0.0015,
    "sigma0": 0.8,
    "eta": 1.2,
    "kappa": 0.02,
}

PARAM_GRIDS = {
    "alpha":  [1.0, 1.5, 2.0, 2.5, 3.0],
    "delta":  [0.0, 0.00075, 0.0015, 0.00225, 0.0030],
    "sigma0": [0.4, 0.6, 0.8, 1.0, 1.2],
    "eta":    [0.4, 0.8, 1.2, 1.6, 2.0],
    "kappa":  [0.00, 0.01, 0.02, 0.03, 0.04],
}


def _find_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_train_test_csvs(dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    dataset_dir = Path(dataset_dir)
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv under {dataset_dir}, got {train_path} and {test_path}"
        )

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

    if not feature_cols:
        raise ValueError("No shared numeric spectral feature columns were found.")

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


class SensitivityDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        humidity_col: str,
        label_to_idx: Dict[str, int],
        param_name: str,
        param_value: float,
    ):
        self.X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).copy()
        self.h = pd.to_numeric(df[humidity_col], errors="coerce").to_numpy(dtype=np.float32).copy()
        labels = df[label_col].astype(str).tolist()
        self.y = np.array([label_to_idx[x] for x in labels], dtype=np.int64)

        self.param_name = param_name
        self.param_value = float(param_value)
        self.params = dict(DEFAULT_PARAMS)
        if param_name not in self.params:
            raise ValueError(f"Unsupported param_name: {param_name}")
        self.params[param_name] = float(param_value)

        self.wl = wavelength_grid(self.X.shape[1])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        spec = apply_humidity_perturbation(
            spectrum=self.X[idx],
            humidity=float(self.h[idx]),
            wl=self.wl,
            alpha=self.params["alpha"],
            delta=self.params["delta"],
            sigma0=self.params["sigma0"],
            eta=self.params["eta"],
            kappa=self.params["kappa"],
        )
        return {
            "spectrum": torch.tensor(spec, dtype=torch.float32),
            "humidity": torch.tensor(self.h[idx], dtype=torch.float32),
            "label": torch.tensor(self.y[idx], dtype=torch.long),
        }
