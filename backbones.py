"""ResNet50 backbone for the altitude encoder."""
from torch import nn
from torchvision.models import resnet50


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.avgpool = None
        self.model.fc = None
        self.model.layer4 = None
        self.out_channels = 1024

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        return self.model.layer3(x)
