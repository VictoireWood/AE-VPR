import math
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

class FootprintMarginClassifier(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m_min=0.0, m_max=0.4):
        super().__init__()
        if not isinstance(in_features, int) or not isinstance(out_features, int) or in_features <= 0 or (out_features <= 0):
            raise ValueError('classifier dimensions must be positive integers')
        if not all((math.isfinite(v) for v in (s, m_min, m_max))):
            raise ValueError('scale and margins must be finite')
        if s <= 0 or not 0 <= m_min <= m_max:
            raise ValueError('require s > 0 and 0 <= m_min <= m_max')
        self.in_features, self.out_features = (in_features, out_features)
        self.s, self.m_min, self.m_max = (float(s), float(m_min), float(m_max))
        self.weight = Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None, footprint_iou=None):
        if input.ndim != 2 or input.shape[1] != self.in_features:
            raise ValueError('input must have shape [B, in_features]')
        dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=input.device.type, enabled=False):
            cosine = F.linear(F.normalize(input.to(dtype), dim=1), F.normalize(self.weight.to(dtype), dim=1))
            if label is None:
                if footprint_iou is not None:
                    raise ValueError('footprint_iou requires training labels')
                return (self.s * cosine, cosine)
            if label.shape != (input.shape[0],) or label.dtype != torch.long or label.device != input.device:
                raise ValueError('label must be int64 [B] on the feature device')
            if ((label < 0) | (label >= self.out_features)).any():
                raise ValueError('label is outside the prototype range')
            if footprint_iou is None or footprint_iou.shape != cosine.shape:
                raise ValueError('footprint_iou must have shape [B, out_features]')
            iou = footprint_iou.detach().to(device=input.device, dtype=dtype)
            if not (torch.isfinite(iou) & (iou >= 0) & (iou <= 1)).all():
                raise ValueError('footprint_iou must be finite and in [0, 1]')
            margins = self.m_min + (self.m_max - self.m_min) * (1 - iou)
            margins = margins.scatter(1, label[:, None], 0.0)
            return (self.s * (cosine + margins), cosine)

class AAMC(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > 0, phi, cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return (output, cosine)
