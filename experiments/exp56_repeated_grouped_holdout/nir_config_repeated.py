import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = {
    "data_sources": [
        {
            "name": "splib07a",
            "base_path": os.path.join(
                PROJECT_ROOT, "usgs_splib07", "ASCIIData", "ASCIIdata_splib07a", "ChapterM_Minerals"
            ),
            "wavelength_file": os.path.join(
                PROJECT_ROOT, "usgs_splib07", "ASCIIData", "ASCIIdata_splib07a",
                "splib07a_Wavelengths_AVIRIS_1996_0.37-2.5_microns.txt"
            ),
        },
        {
            "name": "splib07b",
            "base_path": os.path.join(
                PROJECT_ROOT, "usgs_splib07", "ASCIIData", "ASCIIdata_splib07b", "ChapterM_Minerals"
            ),
            "wavelength_file": os.path.join(
                PROJECT_ROOT, "usgs_splib07", "ASCIIData", "ASCIIdata_splib07b",
                "splib07b_Wavelengths_AVIRIS_1996_interp_to_2203ch.txt"
            ),
        },
    ],
    "output_dir": os.path.join(PROJECT_ROOT, "Dataset", "NIR_repeated_holdout"),
    "minerals": {
        "calcite": 0,
        "azurite": 1,
        "goethite": 2,
        "malachite": 3,
    },
    "train_humidities": np.array([0.0, 0.2, 0.4, 0.6, 0.8]),
    "test_humidities": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
    "wavelength_range": (0.35, 2.5),
    "random_seed": 42,
    "outer_seeds": [42, 52, 62, 72, 82, 92, 102, 112, 122, 132],
    "inner_folds": 4,
    "split": {
        "test_quota_per_class": {
            0: 3,
            1: 3,
            2: 5,
            3: 3,
        }
    },
    "train_augment_times": 10,
    "test_repeat_times": 10,
    "humidity_model": {
        "alpha": 2.0,
        "shift": 0.0015,
        "sigma0": 0.8,
        "sigma_h": 1.2,
        "baseline_k": 0.02,
    },
    "augmentation": {
        "scale": (0.98, 1.02),
        "shift": (-0.001, 0.001),
        "slope": (-0.01, 0.01),
        "noise_std": (0.002, 0.008),
    },
}
