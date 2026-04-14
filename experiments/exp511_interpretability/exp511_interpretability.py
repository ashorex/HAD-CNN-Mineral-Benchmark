from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


LABEL_CANDIDATES = ["label", "target", "class", "mineral", "mineral_class", "y"]
HUMIDITY_CANDIDATES = ["humidity", "rh", "h", "humidity_label"]
GROUP_CANDIDATES = ["source_file", "source_id", "source", "parent_id", "file_id", "group"]

MODEL_SPECS = {
    "had": {
        "module": "Models.HDA_CNN",
        "class": "HDA_CNN",
        "uses_humidity": True,
    },
    "concat": {
        "module": "Models.Concat_CNN",
        "class": "Concat_CNN",
        "uses_humidity": True,
    },
    "cnn": {
        "module": "Models.CNN_1D",
        "class": "CNN_1D",
        "uses_humidity": False,
    },
    "resnet": {
        "module": "Models.ResNet1D",
        "class": "ResNet1D",
        "uses_humidity": False,
    },
    "resnet1d": {
        "module": "Models.ResNet1D",
        "class": "ResNet1D",
        "uses_humidity": False,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_cols: List[str]
    label_col: str
    humidity_col: str
    group_col: Optional[str]
    wavelengths_um: np.ndarray
    labels_sorted: List[str]
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]


def find_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_wavelengths(feature_cols: List[str], wavelengths_csv: Optional[str]) -> np.ndarray:
    if wavelengths_csv:
        path = Path(wavelengths_csv)
        if not path.exists():
            raise FileNotFoundError(f"wavelengths_csv not found: {path}")
        arr = pd.read_csv(path, header=None).values.squeeze()
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or len(arr) != len(feature_cols):
            raise ValueError(
                f"Wavelength axis length mismatch: expected {len(feature_cols)}, got {len(arr)}"
            )
        return arr

    # Try parsing wavelength-like numbers directly from feature names.
    parsed = []
    ok = True
    for col in feature_cols:
        try:
            parsed.append(float(str(col).replace("um", "").replace("μm", "")))
        except Exception:
            ok = False
            break
    if ok:
        arr = np.asarray(parsed, dtype=float)
        if np.all(np.diff(arr) > 0):
            # Heuristically convert from nm to um when needed.
            if arr.max() > 50:
                arr = arr / 1000.0
            return arr

    # Fallback: use the wavelength range described in the manuscript.
    return np.linspace(0.35, 2.50, len(feature_cols), dtype=float)



def load_dataset(dataset_dir: str, split: str, wavelengths_csv: Optional[str]) -> DatasetBundle:
    csv_path = Path(dataset_dir) / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {csv_path}")

    df = pd.read_csv(csv_path)

    label_col = find_col(df.columns, LABEL_CANDIDATES)
    humidity_col = find_col(df.columns, HUMIDITY_CANDIDATES)
    group_col = find_col(df.columns, GROUP_CANDIDATES)

    if label_col is None:
        raise ValueError("Could not identify the label column in the CSV file.")
    if humidity_col is None:
        raise ValueError("Could not identify the humidity column in the CSV file.")

    exclude = {label_col, humidity_col}
    if group_col is not None:
        exclude.add(group_col)

    feature_cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().all():
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No usable spectral feature columns found in the CSV file.")

    wavelengths_um = load_wavelengths(feature_cols, wavelengths_csv)

    labels_sorted = sorted(df[label_col].astype(str).unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels_sorted)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    return DatasetBundle(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        humidity_col=humidity_col,
        group_col=group_col,
        wavelengths_um=wavelengths_um,
        labels_sorted=labels_sorted,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
    )


class SpectralDataset(torch.utils.data.Dataset):
    def __init__(self, bundle: DatasetBundle):
        self.df = bundle.df.reset_index(drop=True).copy()
        self.feature_cols = bundle.feature_cols
        self.label_col = bundle.label_col
        self.humidity_col = bundle.humidity_col
        self.group_col = bundle.group_col
        self.label_to_idx = bundle.label_to_idx
        self.idx_to_label = bundle.idx_to_label
        self.X = self.df[self.feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.h = self.df[self.humidity_col].astype(float).to_numpy(dtype=np.float32)
        self.y = np.array([self.label_to_idx[x] for x in self.df[self.label_col].astype(str)], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        item = {
            "x": torch.tensor(self.X[idx], dtype=torch.float32),
            "h": torch.tensor(self.h[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "meta_index": idx,
        }
        return item


def parse_checkpoint_specs(values: Sequence[str]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                "Each --checkpoint entry must look like model_key=/path/to/checkpoint.pth"
            )
        model_key, path = value.split("=", 1)
        model_key = model_key.strip().lower()
        ckpt = Path(path.strip())
        if model_key not in MODEL_SPECS:
            raise ValueError(f"Unsupported model key: {model_key}")
        out.append((model_key, ckpt))
    return out



def add_project_root_to_path() -> None:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))



def load_model_class(model_key: str):
    spec = MODEL_SPECS[model_key]
    module = importlib.import_module(spec["module"])
    cls = getattr(module, spec["class"])
    return cls



def try_build_model(cls, num_classes: int, input_len: int):
    constructor_attempts = [
        {"num_classes": num_classes, "input_dim": input_len},
        {"num_classes": num_classes, "input_length": input_len},
        {"num_classes": num_classes},
        {"n_classes": num_classes, "input_dim": input_len},
        {"n_classes": num_classes},
        {"classes": num_classes, "input_dim": input_len},
        {"classes": num_classes},
        {"input_dim": input_len, "num_classes": num_classes, "humidity_dim": 1},
        {},
    ]
    errors = []
    for kwargs in constructor_attempts:
        try:
            return cls(**kwargs)
        except Exception as exc:
            errors.append(f"kwargs={kwargs}: {exc}")

    # positional fallbacks
    positional_attempts = [
        (input_len, num_classes),
        (num_classes,),
    ]
    for args in positional_attempts:
        try:
            return cls(*args)
        except Exception as exc:
            errors.append(f"args={args}: {exc}")

    joined = "\n".join(errors[:8])
    raise RuntimeError(
        "Could not instantiate the model. Please adjust try_build_model() to your local "
        f"constructor signature. Attempts:\n{joined}"
    )



def load_checkpoint_into_model(model: nn.Module, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = None
    if isinstance(obj, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                state_dict = obj[key]
                break
        if state_dict is None and all(isinstance(k, str) for k in obj.keys()):
            state_dict = obj
    else:
        state_dict = obj

    if state_dict is None:
        raise RuntimeError("Could not extract a state_dict from the checkpoint file.")

    # Strip possible DistributedDataParallel prefix.
    cleaned = {}
    for k, v in state_dict.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)



def make_input_variant(x: torch.Tensor, shape_mode: str) -> torch.Tensor:
    if shape_mode == "bl":
        return x
    if shape_mode == "b1l":
        return x.unsqueeze(1)
    raise ValueError(shape_mode)



def safe_forward(model: nn.Module, model_key: str, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    uses_humidity = MODEL_SPECS[model_key]["uses_humidity"]
    errors: List[str] = []
    for shape_mode in ["b1l", "bl"]:
        x_in = make_input_variant(x, shape_mode)
        if uses_humidity:
            humidity_variants = [
                h,
                h.unsqueeze(1),
            ]
            for hv in humidity_variants:
                try:
                    return model(x_in, hv)
                except Exception as exc:
                    errors.append(f"shape={shape_mode}, humidity_shape={tuple(hv.shape)}: {exc}")
        else:
            try:
                return model(x_in)
            except Exception as exc:
                errors.append(f"shape={shape_mode}: {exc}")

    joined = "\n".join(errors[:8])
    raise RuntimeError(
        f"Could not run a forward pass for model_key={model_key}. "
        f"Please adjust safe_forward() to your local forward signature. Attempts:\n{joined}"
    )



def integrated_gradients(
    model: nn.Module,
    model_key: str,
    x: torch.Tensor,
    h: torch.Tensor,
    target_idx: int,
    steps: int,
) -> torch.Tensor:
    baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)
    for alpha in torch.linspace(0.0, 1.0, steps, device=x.device):
        x_step = baseline + alpha * (x - baseline)
        x_step = x_step.clone().detach().requires_grad_(True)
        logits = safe_forward(model, model_key, x_step, h)
        score = logits[:, target_idx].sum()
        model.zero_grad(set_to_none=True)
        if x_step.grad is not None:
            x_step.grad.zero_()
        score.backward()
        total_grad += x_step.grad.detach()
    avg_grad = total_grad / float(steps)
    ig = (x - baseline) * avg_grad
    return ig.detach()



def gradient_saliency(
    model: nn.Module,
    model_key: str,
    x: torch.Tensor,
    h: torch.Tensor,
    target_idx: int,
) -> torch.Tensor:
    x_in = x.clone().detach().requires_grad_(True)
    logits = safe_forward(model, model_key, x_in, h)
    score = logits[:, target_idx].sum()
    model.zero_grad(set_to_none=True)
    if x_in.grad is not None:
        x_in.grad.zero_()
    score.backward()
    return x_in.grad.detach()



def normalize_attr(attr: np.ndarray) -> np.ndarray:
    attr = np.abs(attr.astype(np.float64))
    s = attr.sum()
    if s <= 0:
        return np.zeros_like(attr)
    return attr / s



def parse_band_defs(text: Optional[str]) -> List[Tuple[str, float, float]]:
    if not text:
        return [
            ("water_1p4", 1.35, 1.45),
            ("water_1p9", 1.85, 1.95),
            ("mineral_diag", 2.15, 2.35),
        ]
    out = []
    for item in text.split(","):
        name, start, end = item.split(":")
        out.append((name.strip(), float(start), float(end)))
    return out



def band_mass(norm_attr: np.ndarray, wavelengths_um: np.ndarray, bands: List[Tuple[str, float, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, start, end in bands:
        mask = (wavelengths_um >= start) & (wavelengths_um <= end)
        out[name] = float(norm_attr[mask].sum())
    covered = sum(out.values())
    out["other"] = max(0.0, 1.0 - covered)
    return out



def humidity_group(value: float, threshold: float) -> str:
    return "low" if value <= threshold else "high"



def choose_target(pred_idx: int, true_idx: int, target_mode: str) -> int:
    if target_mode == "pred":
        return pred_idx
    if target_mode == "true":
        return true_idx
    raise ValueError(target_mode)



def run_analysis(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    add_project_root_to_path()

    bundle = load_dataset(args.dataset_dir, args.split, args.wavelengths_csv)
    dataset = SpectralDataset(bundle)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    bands = parse_band_defs(args.band_defs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = parse_checkpoint_specs(args.checkpoint)

    sample_rows: List[Dict] = []
    curve_rows: List[Dict] = []
    band_rows: List[Dict] = []
    peak_rows: List[Dict] = []

    X_all = torch.tensor(dataset.X, dtype=torch.float32, device=device)
    H_all = torch.tensor(dataset.h, dtype=torch.float32, device=device)
    Y_all = torch.tensor(dataset.y, dtype=torch.long, device=device)

    groups = np.array([humidity_group(v, args.humidity_threshold) for v in dataset.h], dtype=object)

    for model_key, ckpt_path in checkpoints:
        cls = load_model_class(model_key)
        model = try_build_model(cls, num_classes=len(bundle.labels_sorted), input_len=len(bundle.feature_cols))
        load_checkpoint_into_model(model, ckpt_path)
        model = model.to(device).eval()

        with torch.no_grad():
            logits = safe_forward(model, model_key, X_all, H_all)
            preds = logits.argmax(dim=1)

        per_group_indices: Dict[Tuple[str, str], List[int]] = {}
        for idx in range(len(dataset)):
            cls_name = bundle.idx_to_label[int(dataset.y[idx])]
            hg = groups[idx]
            key = (cls_name, hg)
            ok = True
            if args.only_correct and int(preds[idx].item()) != int(Y_all[idx].item()):
                ok = False
            if ok:
                per_group_indices.setdefault(key, []).append(idx)

        # downsample for speed if requested
        rng = random.Random(args.seed)
        for key, idxs in list(per_group_indices.items()):
            if args.max_samples_per_group and len(idxs) > args.max_samples_per_group:
                rng.shuffle(idxs)
                per_group_indices[key] = sorted(idxs[: args.max_samples_per_group])

        attr_store: Dict[Tuple[str, str], List[np.ndarray]] = {}
        attr_store_global: Dict[str, List[np.ndarray]] = {"low": [], "high": []}

        for key, idxs in per_group_indices.items():
            class_name, hg = key
            for idx in idxs:
                x = X_all[idx: idx + 1]
                h = H_all[idx: idx + 1]
                y_true = int(Y_all[idx].item())
                y_pred = int(preds[idx].item())
                target_idx = choose_target(y_pred, y_true, args.target_mode)

                if args.method == "ig":
                    attr = integrated_gradients(model, model_key, x, h, target_idx, args.ig_steps)
                else:
                    attr = gradient_saliency(model, model_key, x, h, target_idx)
                attr_np = normalize_attr(attr.squeeze(0).detach().cpu().numpy())
                attr_store.setdefault(key, []).append(attr_np)
                attr_store_global[hg].append(attr_np)

                masses = band_mass(attr_np, bundle.wavelengths_um, bands)
                for band_name, mass in masses.items():
                    band_rows.append(
                        {
                            "model": model_key,
                            "checkpoint": str(ckpt_path),
                            "class_name": class_name,
                            "humidity_group": hg,
                            "sample_index": idx,
                            "band_name": band_name,
                            "mass_ratio": mass,
                        }
                    )

                sample_rows.append(
                    {
                        "model": model_key,
                        "checkpoint": str(ckpt_path),
                        "sample_index": idx,
                        "true_label": bundle.idx_to_label[y_true],
                        "pred_label": bundle.idx_to_label[y_pred],
                        "correct": int(y_true == y_pred),
                        "humidity": float(dataset.h[idx]),
                        "humidity_group": hg,
                        "target_label": bundle.idx_to_label[target_idx],
                        "method": args.method,
                    }
                )

        # aggregate global low/high curves
        for hg, arrays in attr_store_global.items():
            if not arrays:
                continue
            stack = np.stack(arrays, axis=0)
            mean_curve = stack.mean(axis=0)
            std_curve = stack.std(axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(mean_curve)
            for wl, mean_v, std_v in zip(bundle.wavelengths_um, mean_curve, std_curve):
                curve_rows.append(
                    {
                        "model": model_key,
                        "checkpoint": str(ckpt_path),
                        "class_name": "ALL",
                        "humidity_group": hg,
                        "wavelength_um": float(wl),
                        "mean_attr": float(mean_v),
                        "std_attr": float(std_v),
                        "n_samples": int(len(stack)),
                    }
                )
            top_idx = np.argsort(-mean_curve)[: args.top_k]
            for rank, i in enumerate(top_idx, start=1):
                peak_rows.append(
                    {
                        "model": model_key,
                        "checkpoint": str(ckpt_path),
                        "class_name": "ALL",
                        "humidity_group": hg,
                        "rank": rank,
                        "wavelength_um": float(bundle.wavelengths_um[i]),
                        "mean_attr": float(mean_curve[i]),
                    }
                )

        # aggregate per-class low/high curves
        for (class_name, hg), arrays in attr_store.items():
            stack = np.stack(arrays, axis=0)
            mean_curve = stack.mean(axis=0)
            std_curve = stack.std(axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(mean_curve)
            for wl, mean_v, std_v in zip(bundle.wavelengths_um, mean_curve, std_curve):
                curve_rows.append(
                    {
                        "model": model_key,
                        "checkpoint": str(ckpt_path),
                        "class_name": class_name,
                        "humidity_group": hg,
                        "wavelength_um": float(wl),
                        "mean_attr": float(mean_v),
                        "std_attr": float(std_v),
                        "n_samples": int(len(stack)),
                    }
                )

    pd.DataFrame(sample_rows).to_csv(output_dir / "exp511_sample_prediction_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve_rows).to_csv(output_dir / "exp511_attribution_curve_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(band_rows).to_csv(output_dir / "exp511_band_mass_sample_level.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(peak_rows).to_csv(output_dir / "exp511_top_peaks_summary.csv", index=False, encoding="utf-8-sig")

    config = vars(args).copy()
    config["bands"] = bands
    with open(output_dir / "exp511_run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("=" * 88)
    print("Experiment 5.11 interpretability analysis finished.")
    print(f"Saved: {output_dir / 'exp511_sample_prediction_log.csv'}")
    print(f"Saved: {output_dir / 'exp511_attribution_curve_summary.csv'}")
    print(f"Saved: {output_dir / 'exp511_band_mass_sample_level.csv'}")
    print(f"Saved: {output_dir / 'exp511_top_peaks_summary.csv'}")
    print(f"Saved: {output_dir / 'exp511_run_config.json'}")
    print("=" * 88)



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 5.11: model interpretability under humidity perturbation"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="Dataset/NIR_cross_humidity",
        help="Directory containing train.csv and test.csv",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to analyze; use test for the paper by default",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeated argument in the form model_key=/path/to/checkpoint.pth . "
             "Example: --checkpoint had=checkpoints/nir_crosshum_v2/hda_nir_seed72.pth",
    )
    parser.add_argument(
        "--wavelengths_csv",
        type=str,
        default=None,
        help="Optional CSV file containing the wavelength axis. If omitted, the script tries to infer it.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/exp511_interpretability/raw",
        help="Directory for raw interpretability outputs",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ig", "grad"],
        default="ig",
        help="Interpretability method: Integrated Gradients (ig) or vanilla gradient saliency (grad)",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=["pred", "true"],
        default="pred",
        help="Attribution target: predicted class or true class",
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=32,
        help="Number of interpolation steps for Integrated Gradients",
    )
    parser.add_argument(
        "--humidity_threshold",
        type=float,
        default=0.5,
        help="Threshold used to split samples into low/high humidity groups",
    )
    parser.add_argument(
        "--band_defs",
        type=str,
        default=None,
        help="Optional custom band definition string such as water_1p4:1.35:1.45,water_1p9:1.85:1.95,mineral_diag:2.15:2.35",
    )
    parser.add_argument(
        "--only_correct",
        action="store_true",
        help="Analyze only correctly classified samples",
    )
    parser.add_argument(
        "--max_samples_per_group",
        type=int,
        default=200,
        help="Maximum number of samples per (class, humidity_group, model) to analyze",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top wavelengths to record from aggregated attribution curves",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser


if __name__ == "__main__":
    run_analysis(build_argparser().parse_args())
