import torch
import torch.nn as nn
import torchvision

# from models.helper import get_pretrained_torchvision_model, get_randomized_torchvision_model
import timm
import numpy as np

ENV2_ARCHS = {
    'efficientnet_v2_s': 1280,
    'efficientnet_v2_m': 1280,
    'efficientnet_v2_l': 1280,
}


def get_pretrained_torchvision_model(backbone_name):
    """This function takes the name of a backbone and returns the pretrained model from torchvision.
    Examples of backbone_name are 'ResNet18' or 'EfficientNet_B0'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.IMAGENET1K_V1)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model

def get_randomized_torchvision_model(backbone_name):
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=None)    # 随机初始化
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=False)
    return model

class EfficientNet_V2(nn.Module):
    def __init__(
            self,
            model_name='efficientnet_v2_m',
            pretrained=True,
            layers_to_freeze=4,
            layers_to_crop = [],
        ):
        super().__init__()  # NOTE - 调用nn.Module的初始化方法

        
        assert model_name in ENV2_ARCHS.keys(), f'Unknown model name {model_name}'
        if pretrained:
            self.model = get_pretrained_torchvision_model(model_name)
        else:
            self.model = get_randomized_torchvision_model(model_name)
        # self.model = timm.create_model(model_name=model_name, pretrained=pretrained)

        self.num_channels = ENV2_ARCHS[model_name]
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # for name, child in self.model.features.named_children():
            #     print(name)

            # for name, child in self.model.features.named_children():
            #     if name == "5":
            #         break
            #     for params in child.parameters():
            #         params.requires_grad = False
            #         if pretrained:
            # if layers_to_freeze >= 0:
            #     self.model.conv_stem.requires_grad_(False)
            #     self.model.blocks[0].requires_grad_(False)
            #     self.model.blocks[1].requires_grad_(False)
            # if layers_to_freeze >= 1:
            #     self.model.blocks[2].requires_grad_(False)
            # if layers_to_freeze >= 2:
            #     self.model.blocks[3].requires_grad_(False)
            # if layers_to_freeze >= 3:
            #     self.model.blocks[4].requires_grad_(False)
            # if layers_to_freeze >= 4:
            #     self.model.blocks[5].requires_grad_(False)
            for name, child in self.model.features.named_children():
                # logging.debug("Freeze all EfficientNet layers up to n.5")
                if name == str(self.layers_to_freeze):
                    break
                for params in child.parameters():
                    params.requires_grad = False

        self.model.global_pool = None
        self.model.fc = None

        if len(layers_to_crop) > 0:
            layers_to_crop_num = layers_to_crop[0]
            self.model = self.model.features[:-layers_to_crop_num]
        else:
            # layers_to_crop_num = 0
            self.model = self.model.features
        # self.model.to('cuda')

    
    # @autocast(True)
    def forward(self, x: torch.Tensor):
        # x = self.model.features(x)
        # with torch.autocast('cuda'):
        #     x = self.model(x)
        x = self.model(x)
        return x
        
    
    # def output_shape(self):
    #     torch.randn(1, 3, 360, 480)
    #     m = EfficientNet_V2(model_name='efficientnet_v2_m',
    #                     pretrained=True,
    #                     layers_to_freeze=0,)
    #     r = m(x)
    #     return r.shape



def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')

if __name__ == '__main__':
    x = torch.randn(32, 3, 360, 480)
    m = EfficientNet_V2(model_name='efficientnet_v2_m',
                        pretrained=True,
                        layers_to_freeze=7,
                        layers_to_crop = [],
                        )
    r = m(x)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')