import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import random

# ==============================
# 真实水吸收模型
# ==============================
def physical_water_absorption(wavelengths):

    peaks = [0.97, 1.19, 1.45, 1.94]
    absorption = np.zeros_like(wavelengths)

    for p in peaks:
        absorption += np.exp(-(wavelengths - p)**2 / (2 * 0.015**2))

    absorption /= absorption.max()
    return absorption


# ==============================
# 湿度模拟模型
# ==============================
#实验一使用的最强扰动配置
# def simulate_humidity_advanced(spectrum, wavelengths, humidity, augment=False):
#
#     water_abs = physical_water_absorption(wavelengths)
#
#     # 湿度吸收
#     spec = spectrum * np.exp(-humidity * 2.5 * water_abs)
#
#     # 峰位红移
#     shift = humidity * 0.003
#     shifted_wave = wavelengths + shift
#
#     interp = interp1d(
#         shifted_wave,
#         spec,
#         bounds_error=False,
#         fill_value="extrapolate"
#     )
#     spec = interp(wavelengths)
#
#     # 峰展宽
#     sigma = 1 + humidity * 2
#     spec = gaussian_filter1d(spec, sigma=sigma)
#
#     # 基线漂移
#     baseline = humidity * 0.05 * (wavelengths - wavelengths.min())
#     spec = spec + baseline
#
#     if augment:
#
#         # 强度缩放
#         scale = np.random.uniform(0.95, 1.05)
#         spec = spec * scale
#
#         # 波长随机偏移
#         rand_shift = np.random.uniform(-0.002, 0.002)
#         interp = interp1d(
#             wavelengths + rand_shift,
#             spec,
#             bounds_error=False,
#             fill_value="extrapolate"
#         )
#         spec = interp(wavelengths)
#
#         # 随机基线
#         slope = np.random.uniform(-0.02, 0.02)
#         spec = spec + slope * (wavelengths - wavelengths.min())
#
#         # 噪声
#         noise_std = np.random.uniform(0.005, 0.02)
#         noise = np.random.normal(0, noise_std, size=spec.shape)
#         spec = spec + noise
#
#     return spec


#实验二使用的扰动：
def simulate_humidity_advanced(spectrum, wavelengths, humidity, augment=False):

    water_abs = physical_water_absorption(wavelengths)

    # =========================
    # 1. 湿度相关物理扰动
    # =========================

    # 湿度吸收（略微减弱强度，避免过强压制原始谱形）
    spec = spectrum * np.exp(-humidity * 2.0 * water_abs)

    # 峰位红移（减小偏移幅度）
    shift = humidity * 0.0015
    shifted_wave = wavelengths + shift

    interp = interp1d(
        shifted_wave,
        spec,
        bounds_error=False,
        fill_value="extrapolate"
    )
    spec = interp(wavelengths)

    # 峰展宽（保留，但适当减弱）
    sigma = 0.8 + humidity * 1.2
    spec = gaussian_filter1d(spec, sigma=sigma)

    # 基线漂移（减弱）
    baseline = humidity * 0.02 * (wavelengths - wavelengths.min())
    spec = spec + baseline

    # =========================
    # 2. 训练阶段随机增强
    # =========================
    if augment:

        # 强度缩放：缩小扰动范围
        scale = np.random.uniform(0.98, 1.02)
        spec = spec * scale

        # 波长随机偏移：减小偏移量
        rand_shift = np.random.uniform(-0.001, 0.001)
        interp = interp1d(
            wavelengths + rand_shift,
            spec,
            bounds_error=False,
            fill_value="extrapolate"
        )
        spec = interp(wavelengths)

        # 随机基线：减弱斜率范围
        slope = np.random.uniform(-0.01, 0.01)
        spec = spec + slope * (wavelengths - wavelengths.min())

        # 噪声：减弱噪声强度
        noise_std = np.random.uniform(0.002, 0.008)
        noise = np.random.normal(0, noise_std, size=spec.shape)
        spec = spec + noise

    return spec
# ==============================
# 主构建函数
# ==============================
def build_dataset(config):

    DATA_SOURCES = config["data_sources"]
    OUTPUT_DIR = config["output_dir"]
    MINERALS = config["minerals"]
    TRAIN_HUMIDITIES = config["train_humidities"]
    TEST_HUMIDITY = config["test_humidity"]
    WAVELENGTH_RANGE = config["wavelength_range"]
    TRAIN_RATIO = config.get("train_ratio", 0.8)
    RANDOM_SEED = config.get("random_seed", 42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def load_wavelengths(path):
        wavelengths = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                try:
                    wavelengths.append(float(line))
                except:
                    continue
        return np.array(wavelengths)

    def load_and_preprocess_spectrum(file_path, wavelengths, mask):
        spectrum = np.loadtxt(file_path, skiprows=1, dtype=float)
        spectrum = np.nan_to_num(spectrum)

        if spectrum.ndim == 2:
            spectrum = spectrum[:, -1]

        if len(spectrum) != len(wavelengths):
            x_old = np.linspace(0, 1, len(spectrum))
            x_new = np.linspace(0, 1, len(wavelengths))
            interp = interp1d(x_old, spectrum, kind="linear")
            spectrum = interp(x_new)

        spectrum = spectrum[mask]
        spectrum = savgol_filter(spectrum, 11, 3)
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)

        return spectrum

    # =========================================================
    # 1. 建立统一公共波长轴（默认采用第一个数据源的波长轴）
    # =========================================================
    ref_wavelengths = load_wavelengths(DATA_SOURCES[0]["wavelength_file"])
    ref_mask = (ref_wavelengths >= WAVELENGTH_RANGE[0]) & (ref_wavelengths <= WAVELENGTH_RANGE[1])
    used_wavelengths = ref_wavelengths[ref_mask]

    # =========================================================
    # 2. 收集所有原始文件，并按类别分组
    # =========================================================
    class_files = {label: [] for _, label in MINERALS.items()}

    for source in DATA_SOURCES:
        base_path = source["base_path"]

        for root, _, files in os.walk(base_path):
            for file in files:
                if not file.endswith(".txt"):
                    continue

                fname_lower = file.lower()
                file_path = os.path.join(root, file)

                for mineral, label in MINERALS.items():
                    if mineral in fname_lower:
                        # 为避免 07a 和 07b 同名文件冲突，把相对路径作为唯一标识来源
                        class_files[label].append(file_path)
                        print("匹配:", mineral, "->", file_path)
                        break

    print("\n===== 各类别原始文件数量 =====")
    for label, file_list in class_files.items():
        print(f"Class {label}: {len(file_list)} files")

    # =========================================================
    # 3. 先按原始文件划分 train/test
    # =========================================================
    random.seed(RANDOM_SEED)

    train_source_files = []
    test_source_files = []

    for label, file_list in class_files.items():
        file_list = file_list.copy()
        random.shuffle(file_list)

        n_total = len(file_list)
        if n_total == 0:
            continue

        n_train = int(n_total * TRAIN_RATIO)

        if n_total >= 2:
            n_train = max(1, min(n_train, n_total - 1))
        else:
            n_train = 1

        train_part = file_list[:n_train]
        test_part = file_list[n_train:]

        train_source_files.extend([(fp, label) for fp in train_part])
        test_source_files.extend([(fp, label) for fp in test_part])

    print("\n===== 原始文件划分结果 =====")
    print("训练原始文件数:", len(train_source_files))
    print("测试原始文件数:", len(test_source_files))

    # =========================================================
    # 4. 分别生成 train/test 样本
    #    每个 file_path 根据它属于哪个 source，使用对应 wavelength_file
    # =========================================================
    train_rows = []
    test_rows = []

    def get_source_wavelengths(file_path):
        for source in DATA_SOURCES:
            if source["base_path"] in file_path:
                wavelengths = load_wavelengths(source["wavelength_file"])
                mask = (wavelengths >= WAVELENGTH_RANGE[0]) & (wavelengths <= WAVELENGTH_RANGE[1])
                return wavelengths, mask
        raise ValueError(f"无法识别文件属于哪个数据源: {file_path}")

    AUGMENT_TIMES = 10
    TEST_AUGMENT = 10

    # 训练集生成
    for file_path, label in train_source_files:
        try:
            src_wavelengths, src_mask = get_source_wavelengths(file_path)
            spectrum = load_and_preprocess_spectrum(file_path, src_wavelengths, src_mask)

            # 若当前源波长轴与公共波长轴不同，则统一插值到公共波长轴
            src_used_wavelengths = src_wavelengths[src_mask]
            if len(src_used_wavelengths) != len(used_wavelengths) or not np.allclose(src_used_wavelengths, used_wavelengths):
                interp = interp1d(src_used_wavelengths, spectrum, kind="linear", bounds_error=False, fill_value="extrapolate")
                spectrum = interp(used_wavelengths)

            for h in TRAIN_HUMIDITIES:
                for _ in range(AUGMENT_TIMES):
                    wet = simulate_humidity_advanced(
                        spectrum,
                        used_wavelengths,
                        h,
                        augment=True
                    )
                    train_rows.append(
                        [file_path, label, h] + wet.tolist()
                    )
        except Exception as e:
            print("训练样本处理失败:", file_path, e)
            continue

    # 测试集生成
    for file_path, label in test_source_files:
        try:
            src_wavelengths, src_mask = get_source_wavelengths(file_path)
            spectrum = load_and_preprocess_spectrum(file_path, src_wavelengths, src_mask)

            src_used_wavelengths = src_wavelengths[src_mask]
            if len(src_used_wavelengths) != len(used_wavelengths) or not np.allclose(src_used_wavelengths, used_wavelengths):
                interp = interp1d(src_used_wavelengths, spectrum, kind="linear", bounds_error=False, fill_value="extrapolate")
                spectrum = interp(used_wavelengths)

            for h in TEST_HUMIDITY:
                for _ in range(TEST_AUGMENT):
                    wet = simulate_humidity_advanced(
                        spectrum,
                        used_wavelengths,
                        h,
                        augment=False
                    )
                    test_rows.append(
                        [file_path, label, h] + wet.tolist()
                    )
        except Exception as e:
            print("测试样本处理失败:", file_path, e)
            continue

    columns = ["source_file", "label", "humidity"] + [f"{int(w*1000)}nm" for w in used_wavelengths]

    train_df = pd.DataFrame(train_rows, columns=columns)
    test_df = pd.DataFrame(test_rows, columns=columns)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("\n===== 数据集构建完成 =====")
    print("训练样本数:", len(train_df))
    print("测试样本数:", len(test_df))

    train_sources = set(train_df["source_file"].unique())
    test_sources = set(test_df["source_file"].unique())
    overlap = train_sources.intersection(test_sources)

    print("训练集原始文件数:", len(train_sources))
    print("测试集原始文件数:", len(test_sources))
    print("train/test 原始文件重叠数:", len(overlap))
    if len(overlap) > 0:
        print("警告：仍存在原始文件重叠！")
    else:
        print("检查通过：train/test 原始文件完全独立。")