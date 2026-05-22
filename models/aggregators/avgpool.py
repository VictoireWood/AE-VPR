from torch import nn

class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Flatten(),
        )
    def forward(self, x):
        x = self.pooling(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        """Flatten a pooled BCHW tensor into BC."""
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:,:,0,0]
