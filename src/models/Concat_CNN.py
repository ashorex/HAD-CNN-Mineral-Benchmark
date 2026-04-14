import torch
import torch.nn as nn


class HumidityEmbedding(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim)
        )

    def forward(self, h):
        return self.embed(h)


class SpectralBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        feat = self.features(x)
        return feat.view(feat.size(0), -1)   # [B,128]


class Concat_CNN(nn.Module):
    def __init__(self, num_classes=4, hum_dim=16):
        super().__init__()
        self.backbone = SpectralBackbone()
        self.hum_embed = HumidityEmbedding(embed_dim=hum_dim)

        self.classifier = nn.Sequential(
            nn.Linear(128 + hum_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_spec, x_hum):
        spec_feat = self.backbone(x_spec)
        hum_feat = self.hum_embed(x_hum)
        feat = torch.cat([spec_feat, hum_feat], dim=1)
        out = self.classifier(feat)
        return out