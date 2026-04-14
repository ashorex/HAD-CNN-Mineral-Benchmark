import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SpectralDataset(Dataset):
    """
    读取CSV光谱数据
    支持包含 source_file / label / humidity / 光谱特征列 的数据格式
    """

    def __init__(self, csv_path):

        df = pd.read_csv(csv_path)

        # 可选：保存原始文件名，方便后续调试或核查
        if "source_file" in df.columns:
            self.source_files = df["source_file"].values
        else:
            self.source_files = None

        self.labels = df["label"].values.astype(np.int64)
        self.humidity = df["humidity"].values.astype(np.float32)

        # 显式去掉非光谱列，避免列位置变化带来错误
        drop_cols = ["label", "humidity"]
        if "source_file" in df.columns:
            drop_cols.append("source_file")

        self.spectra = df.drop(columns=drop_cols).values.astype(np.float32)

        # reshape 为 [N, 1, L]
        self.spectra = self.spectra[:, np.newaxis, :]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        spec = torch.tensor(self.spectra[idx], dtype=torch.float32)
        hum = torch.tensor([self.humidity[idx]], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return spec, hum, label