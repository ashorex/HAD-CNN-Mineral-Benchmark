import os
import numpy as np

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = {
    # 多数据源目录
    "data_sources": [
        {
            "name": "splib07a",
            "base_path": os.path.join(
                PROJECT_ROOT,
                "usgs_splib07",
                "ASCIIData",
                "ASCIIdata_splib07a",
                "ChapterM_Minerals"
            ),
            "wavelength_file": os.path.join(
                PROJECT_ROOT,
                "usgs_splib07",
                "ASCIIData",
                "ASCIIdata_splib07a",
                "splib07a_Wavelengths_AVIRIS_1996_0.37-2.5_microns.txt"
            )
        },
        {
            "name": "splib07b",
            "base_path": os.path.join(
                PROJECT_ROOT,
                "usgs_splib07",
                "ASCIIData",
                "ASCIIdata_splib07b",
                "ChapterM_Minerals"
            ),
            "wavelength_file": os.path.join(
                PROJECT_ROOT,
                "usgs_splib07",
                "ASCIIData",
                "ASCIIdata_splib07b",
                "splib07b_Wavelengths_AVIRIS_1996_interp_to_2203ch.txt"
            )
        }
    ],

    # NIR数据文件输出目录
    # "output_dir": os.path.join(PROJECT_ROOT, "Dataset", "NIR"),

    #湿度泛化实验数据集输出目录：
    "output_dir": os.path.join(PROJECT_ROOT, "Dataset", "NIR_cross_humidity"),

    # 4类版本
    "minerals": {
        "calcite": 0,
        "azurite": 1,
        "goethite": 2,
        "malachite": 3,
    },

    #常规测试数据集文件湿度设置
    # "train_humidities": np.linspace(0, 1, 30),
    # "test_humidity": np.arange(0.1, 1.0, 0.05),

    #跨湿度泛化数据湿度设置：
    "train_humidities": np.array([0.0, 0.2, 0.4, 0.6, 0.8]),
    "test_humidity": np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
    "wavelength_range": (0.35, 2.5),

    "train_ratio": 0.8,
    "random_seed": 42,

}