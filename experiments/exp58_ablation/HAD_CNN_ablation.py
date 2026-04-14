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
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward_head(self, feat_map):
        feat = self.pool(feat_map)
        feat = feat.view(feat.size(0), -1)
        return feat


class HumidityFeatureMapModulation(nn.Module):
    def __init__(self, channels=128, hum_dim=16, hidden_dim=64):
        super().__init__()
        self.gamma_gen = nn.Sequential(
            nn.Linear(hum_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, channels)
        )
        self.beta_gen = nn.Sequential(
            nn.Linear(hum_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, channels)
        )

    def forward(self, feat_map, hum_feat, use_residual=True):
        gamma = self.gamma_gen(hum_feat).unsqueeze(-1)
        beta = self.beta_gen(hum_feat).unsqueeze(-1)
        gamma = 0.2 * torch.tanh(gamma)
        beta = 0.2 * torch.tanh(beta)
        mod_map = feat_map * (1.0 + gamma) + beta
        if use_residual:
            return feat_map + mod_map
        return mod_map


class HDA_CNN_Ablation(nn.Module):
    def __init__(self, num_classes=4, hum_dim=16, variant="full"):
        super().__init__()
        valid = {"full", "late_concat_only", "modulation_only", "no_residual"}
        if variant not in valid:
            raise ValueError(f"variant must be one of {valid}, got {variant}")
        self.variant = variant

        self.backbone = SpectralBackbone()
        self.hum_embed = HumidityEmbedding(embed_dim=hum_dim)
        self.modulation = HumidityFeatureMapModulation(channels=128, hum_dim=hum_dim, hidden_dim=64)

        if variant == "modulation_only":
            cls_in_dim = 128
        else:
            cls_in_dim = 128 + hum_dim

        self.classifier = nn.Sequential(
            nn.Linear(cls_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_spec, x_hum):
        hum_feat = self.hum_embed(x_hum)
        feat_map = self.backbone.forward_features(x_spec)

        if self.variant == "late_concat_only":
            fused_map = feat_map
        elif self.variant == "no_residual":
            fused_map = self.modulation(feat_map, hum_feat, use_residual=False)
        else:
            fused_map = self.modulation(feat_map, hum_feat, use_residual=True)

        spec_feat = self.backbone.forward_head(fused_map)

        if self.variant == "modulation_only":
            fused_feat = spec_feat
        else:
            fused_feat = torch.cat([spec_feat, hum_feat], dim=1)

        return self.classifier(fused_feat)
