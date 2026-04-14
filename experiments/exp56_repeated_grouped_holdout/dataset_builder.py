import os
import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
def physical_water_absorption(wavelengths):
    peaks = np.array([0.97, 1.19, 1.45, 1.94])
    width = 0.015
    absorption = sum(np.exp(-((wavelengths - p) ** 2) / (2 * width ** 2)) for p in peaks)
    return absorption / (absorption.max() + 1e-12)


def simulate_humidity(spectrum, wavelengths, humidity, cfg, augment=False):
    hcfg = cfg["humidity_model"]
    acfg = cfg["augmentation"]
    water = physical_water_absorption(wavelengths)

    spec = spectrum * np.exp(-humidity * hcfg["alpha"] * water)

    shift = humidity * hcfg["shift"]
    spec = interp1d(
        wavelengths + shift, spec, bounds_error=False, fill_value="extrapolate"
    )(wavelengths)

    sigma = hcfg["sigma0"] + humidity * hcfg["sigma_h"]
    spec = gaussian_filter1d(spec, sigma=sigma)

    spec = spec + humidity * hcfg["baseline_k"] * (wavelengths - wavelengths.min())

    if augment:
        spec *= np.random.uniform(*acfg["scale"])
        rand_shift = np.random.uniform(*acfg["shift"])
        spec = interp1d(
            wavelengths + rand_shift, spec, bounds_error=False, fill_value="extrapolate"
        )(wavelengths)
        spec = spec + np.random.uniform(*acfg["slope"]) * (wavelengths - wavelengths.min())
        spec = spec + np.random.normal(0, np.random.uniform(*acfg["noise_std"]), size=spec.shape)

    return spec

@lru_cache(maxsize=None)
def load_wavelengths(path):
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                vals.append(float(line.strip()))
            except Exception:
                pass
    return np.asarray(vals, dtype=float)


def _safe_savgol(x):
    if len(x) < 5:
        return x
    win = min(11, len(x) if len(x) % 2 == 1 else len(x) - 1)
    return savgol_filter(x, max(win, 5), 3 if win >= 7 else 2)


def preprocess_spectrum(file_path, src_wavelengths, wl_range, target_wavelengths):
    spectrum = np.loadtxt(file_path, skiprows=1, dtype=float)
    spectrum = np.nan_to_num(spectrum[:, -1] if np.ndim(spectrum) == 2 else spectrum)

    if len(spectrum) != len(src_wavelengths):
        spectrum = interp1d(
            np.linspace(0, 1, len(spectrum)),
            spectrum,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(np.linspace(0, 1, len(src_wavelengths)))

    mask = (src_wavelengths >= wl_range[0]) & (src_wavelengths <= wl_range[1])
    spectrum = _safe_savgol(spectrum[mask])
    used_w = src_wavelengths[mask]

    if len(used_w) != len(target_wavelengths) or not np.allclose(used_w, target_wavelengths):
        spectrum = interp1d(
            used_w, spectrum, kind="linear", bounds_error=False, fill_value="extrapolate"
        )(target_wavelengths)

    mn, mx = spectrum.min(), spectrum.max()
    return (spectrum - mn) / (mx - mn + 1e-8)


def get_target_axis(cfg):
    ref_w = load_wavelengths(cfg["data_sources"][0]["wavelength_file"])
    lo, hi = cfg["wavelength_range"]
    return ref_w[(ref_w >= lo) & (ref_w <= hi)]


def match_source(file_path, data_sources):
    file_path = str(Path(file_path).resolve())
    for src in data_sources:
        base = str(Path(src["base_path"]).resolve())
        if os.path.commonpath([file_path, base]) == base:
            return src
    raise ValueError(f"Cannot locate source for: {file_path}")

def collect_class_files(cfg):
    files_by_label = {label: [] for label in cfg["minerals"].values()}
    for src in cfg["data_sources"]:
        for root, _, files in os.walk(src["base_path"]):
            for file in files:
                if not file.endswith(".txt"):
                    continue
                lower = file.lower()
                path = os.path.join(root, file)
                for mineral, label in cfg["minerals"].items():
                    if mineral in lower:
                        files_by_label[label].append(path)
                        break

    for label in files_by_label:
        files_by_label[label] = sorted(set(files_by_label[label]))
    return files_by_label


def fixed_quota_source_holdout(files_by_label, test_quota_per_class, seed):
    rng = random.Random(seed)
    train_pairs, test_pairs = [], []

    for label, files in files_by_label.items():
        files = files.copy()
        rng.shuffle(files)
        n_test = int(test_quota_per_class[label])
        if n_test >= len(files):
            raise ValueError(
                f"Class {label}: requested test quota={n_test}, total files={len(files)}"
            )
        test_files = files[:n_test]
        train_files = files[n_test:]

        train_pairs.extend((fp, label) for fp in train_files)
        test_pairs.extend((fp, label) for fp in test_files)

    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)
    return train_pairs, test_pairs


def grouped_kfold_manifest(train_pairs, n_splits=4, seed=42):
    files_by_label = defaultdict(list)
    for fp, label in train_pairs:
        files_by_label[label].append(fp)

    rng = random.Random(seed)
    folds = [[] for _ in range(n_splits)]
    for label, files in files_by_label.items():
        files = files.copy()
        rng.shuffle(files)
        for i, fp in enumerate(files):
            folds[i % n_splits].append((fp, label))

    manifest = []
    for i in range(n_splits):
        val = folds[i]
        train = [x for j, fold in enumerate(folds) if j != i for x in fold]
        manifest.append(
            {
                "fold": i,
                "train_sources": [fp for fp, _ in train],
                "val_sources": [fp for fp, _ in val],
            }
        )
    return manifest

def build_rows(pairs, humidities, repeats, augment, cfg, target_wavelengths):
    rows, cache = [], {}
    for file_path, label in pairs:
        try:
            if file_path not in cache:
                src = match_source(file_path, cfg["data_sources"])
                src_w = load_wavelengths(src["wavelength_file"])
                cache[file_path] = preprocess_spectrum(
                    file_path, src_w, cfg["wavelength_range"], target_wavelengths
                )

            base_spec = cache[file_path]
            for h in humidities:
                for _ in range(repeats):
                    wet = simulate_humidity(base_spec, target_wavelengths, float(h), cfg, augment=augment)
                    rows.append([file_path, label, float(h), *wet.tolist()])
        except Exception as e:
            print(f"[WARN] skip {file_path}: {e}")
    return rows


def _count_by_label(pairs):
    out = defaultdict(int)
    for _, label in pairs:
        out[int(label)] += 1
    return dict(out)


def build_one_split(cfg, train_pairs, test_pairs, out_dir, split_seed):
    os.makedirs(out_dir, exist_ok=True)
    target_w = get_target_axis(cfg)
    cols = ["source_file", "label", "humidity"] + [f"{int(w*1000)}nm" for w in target_w]

    train_rows = build_rows(
        train_pairs,
        cfg["train_humidities"],
        cfg.get("train_augment_times", 10),
        True,
        cfg,
        target_w,
    )
    test_rows = build_rows(
        test_pairs,
        cfg["test_humidities"],
        cfg.get("test_repeat_times", 10),
        False,
        cfg,
        target_w,
    )

    train_df = pd.DataFrame(train_rows, columns=cols)
    test_df = pd.DataFrame(test_rows, columns=cols)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    meta = {
        "split_seed": split_seed,
        "train_source_count": len({fp for fp, _ in train_pairs}),
        "test_source_count": len({fp for fp, _ in test_pairs}),
        "train_label_counts": _count_by_label(train_pairs),
        "test_label_counts": _count_by_label(test_pairs),
        "overlap": sorted(set(train_df["source_file"].unique()) & set(test_df["source_file"].unique())),
        "inner_cv": grouped_kfold_manifest(
            train_pairs,
            n_splits=cfg.get("inner_folds", 4),
            seed=split_seed + 1000,
        ),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[split={split_seed}] train_sources={meta['train_source_count']} "
        f"test_sources={meta['test_source_count']} "
        f"train_rows={len(train_df)} test_rows={len(test_df)} overlap={len(meta['overlap'])}"
    )
    print(f"             train_counts={meta['train_label_counts']} test_counts={meta['test_label_counts']}")
    return meta


def build_repeated_holdout(cfg):
    out_root = cfg["output_dir"]
    os.makedirs(out_root, exist_ok=True)

    files_by_label = collect_class_files(cfg)
    manifest = []
    seeds = cfg.get("outer_seeds") or [cfg.get("random_seed", 42)]
    test_quota = cfg["split"]["test_quota_per_class"]

    print("===== source-file counts =====")
    for label, files in files_by_label.items():
        print(f"class {label}: {len(files)} files")

    for idx, seed in enumerate(seeds):
        train_pairs, test_pairs = fixed_quota_source_holdout(files_by_label, test_quota, seed)
        split_dir = os.path.join(out_root, f"split_{idx:02d}_seed_{seed}")
        meta = build_one_split(cfg, train_pairs, test_pairs, split_dir, seed)
        manifest.append(
            {
                "split_id": idx,
                "split_seed": seed,
                "split_dir": split_dir,
                "train_source_count": meta["train_source_count"],
                "test_source_count": meta["test_source_count"],
            }
        )

    pd.DataFrame(manifest).to_csv(os.path.join(out_root, "manifest.csv"), index=False)
    print(f"\nDone. Repeated holdout datasets saved to: {out_root}")


if __name__ == "__main__":
    from nir_config_repeated import config
    build_repeated_holdout(config)
