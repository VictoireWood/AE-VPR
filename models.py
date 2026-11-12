"""Model construction and checkpoint loading."""
import torch
from torch import nn
from torchvision.models import efficientnet_b5
from backbones import ResNet
from aggregation import MixVPR

class HeightMixVPR(MixVPR):
    def forward(self, x):
        x = self.mix(x.flatten(2))
        x = self.channel_proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return nn.functional.normalize(self.row_proj(x.float()).flatten(1), dim=-1)

class HeightEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(ResNet(),
                         HeightMixVPR(1024, 21, 28, 256, 4, 1, 4))

class RetrievalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b5(weights=None).features
        self.agg = MixVPR(2048, 7, 7, 768, 4, 1, 4)
    def forward(self, x):
        return self.agg(self.backbone(x))

def load_models(height_checkpoint, retrieval_checkpoint, device='cpu'):
    hstate = torch.load(height_checkpoint, map_location='cpu', weights_only=False)
    rstate = torch.load(retrieval_checkpoint, map_location='cpu', weights_only=False)
    height, retrieval = HeightEncoder(), RetrievalEncoder()
    height.load_state_dict(hstate['model'], strict=True)
    retrieval.load_state_dict(rstate['model'], strict=True)
    weight = hstate['classifier']['weight']
    if weight.shape != (12, 1024):
        raise ValueError('The 50 m altitude classifier weights must have shape [12, 1024]')
    return height.to(device).eval(), weight.to(device), retrieval.to(device).eval()
