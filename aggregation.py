import torch
from torch import nn
from torch.nn import functional as F

class FeatureMixerLayer(nn.Module):

    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, int(in_dim * mlp_ratio)), nn.ReLU(), nn.Linear(int(in_dim * mlp_ratio), in_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)

class MixVPR(nn.Module):

    def __init__(self, in_channels=1024, in_h=20, in_w=20, out_channels=512, mix_depth=1, mlp_ratio=1, out_rows=4) -> None:
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_rows = out_rows
        self.mix_depth = mix_depth
        self.mlp_ratio = mlp_ratio
        hw = in_h * in_w
        self.mix = nn.Sequential(*[FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio) for _ in range(self.mix_depth)])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x
