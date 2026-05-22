import numpy as np
from models import aggregators
from models import backbones
import torchvision
import torch
from torch.nn import Module
import logging

def get_pretrained_torchvision_model(backbone_name):
    """This function takes the name of a backbone and returns the pretrained model from torchvision.
    Examples of backbone_name are 'ResNet18' or 'EfficientNet_B0'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model

# def get_randomized_torchvision_model(backbone_name):
#     try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
#         weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
#         model = getattr(torchvision.models, backbone_name.lower())(weights=None)    # 随机初始化
#     except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
#         model = getattr(torchvision.models, backbone_name.lower())(pretrained=False)
#     return model

def print_nb_params(m, tag: str='total'):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f'{tag}: Trainable parameters: {params/1e6:.4}M')

class HeightFeatureNet(Module):
    def __init__(self, backbone_arch, backbone_info, agg_arch, agg_config):
        super().__init__()
        DINOV2_ARCHS = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536,
            'dinov2_vits14_reg': 384,
            'dinov2_vitb14_reg': 768,
            'dinov2_vitl14_reg': 1024,
            'dinov2_vitg14_reg': 1536,
        }
        input_size = backbone_info['input_size']
        del backbone_info['input_size']
        mixvpr_out_channels = agg_config.pop('mixvpr_out_channels', None)
        self.feature_dim = None
        self.backbone_arch = backbone_arch
        self.backbone_info = backbone_info
        self.backbone = get_backbone(backbone_arch, **backbone_info)

        # Build the aggregator config from the selected backbone feature shape.
        if 'mixvpr' in agg_arch.lower():
            if 'dinov2' in backbone_arch.lower():
                agg_config['in_channels'] = DINOV2_ARCHS[backbone_arch]   
                agg_config['in_h'] = input_size[0] // 14
                agg_config['in_w'] = input_size[1] // 14
            else:
                in_tmp = torch.ones(1, 3, input_size[0], input_size[1])
                out_size = self.backbone(in_tmp).shape
                agg_config['in_channels'] = out_size[1]
                agg_config['in_h'] = out_size[2]
                agg_config['in_w'] = out_size[3]
            agg_config['out_rows'] = 4
            agg_config['mix_depth'] = 4
            if mixvpr_out_channels is not None:
                agg_config['out_channels'] = mixvpr_out_channels
            else:
                agg_config['out_channels'] = 256
            self.feature_dim = agg_config['out_channels'] * agg_config['out_rows']

        elif 'cosplace' in agg_arch.lower():    # assert 'in_dim' 'out_dim' 
            if 'dinov2' in backbone_arch.lower():
                agg_config['in_dim'] = DINOV2_ARCHS[backbone_arch]   
            else:
                in_tmp = torch.ones(1, 3, input_size[0], input_size[1])
                out_size = self.backbone(in_tmp).shape
                agg_config['in_dim'] = out_size[1]
            agg_config['out_dim'] = agg_config['in_dim'] // 2
            self.feature_dim = agg_config['out_dim']

        elif 'convap' in agg_arch.lower():    # assert 'in_channels'
            if 'dinov2' in backbone_arch.lower():
                agg_config['in_channels'] = DINOV2_ARCHS[backbone_arch]
            else:
                in_tmp = torch.ones(1, 3, input_size[0], input_size[1])
                out_size = self.backbone(in_tmp).shape
                agg_config['in_channels'] = out_size[1]
            agg_config['out_channels'] = agg_config['in_channels'] // 2
            self.feature_dim = agg_config['out_channels']

        elif 'salad' in agg_arch.lower():   # assert 'num_channels''num_clusters''cluster_dim''token_dim' in agg_config
            if 'dinov2' in backbone_arch.lower():
                agg_config['num_channels'] = DINOV2_ARCHS[backbone_arch]
            else:
                in_tmp = torch.ones(1, 3, input_size[0], input_size[1])
                out_size = self.backbone(in_tmp).shape
                agg_config['num_channels'] = out_size[1]
            self.feature_dim = agg_config['num_clusters'] * agg_config['cluster_dim'] + agg_config['token_dim']

        elif 'gem' in agg_arch.lower():    # assert 'p'
            if 'dinov2' in backbone_arch.lower():
                self.feature_dim = DINOV2_ARCHS[backbone_arch]
            else:
                in_tmp = torch.ones(1, 3, input_size[0], input_size[1])
                out_size = self.backbone(in_tmp).shape
                self.feature_dim = out_size[1]

        elif 'avgpool' in agg_arch.lower():
            if 'dinov2' in backbone_arch.lower():
                self.feature_dim = DINOV2_ARCHS[backbone_arch]
            else:
                in_tmp = torch.ones(1, 3, input_size[0], input_size[1])
                out_size = self.backbone(in_tmp).shape
                self.feature_dim = out_size[1]
                pass
        
        self.agg_arch = agg_arch
        self.agg_config = agg_config
        # self.aggregator = get_aggregator(agg_arch, **agg_config)
        self.aggregator = get_aggregator(agg_arch, agg_config)
        print_nb_params(self.backbone, 'backbone')
        print_nb_params(self.aggregator, 'aggregator')        
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x


def get_backbone(backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 num_trainable_blocks = 2,
                 ):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where 
                                         we sometimes need to crop the last 
                                         residual block (ex. [4]). Defaults to [].

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if 'resnet' in backbone_arch.lower():
        return backbones.ResNet(backbone_arch.lower(), pretrained, layers_to_freeze, layers_to_crop)
    
    elif 'efficientnet_v2' in backbone_arch.lower():
        return backbones.EfficientNet_V2(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)

    elif 'efficient' in backbone_arch.lower():
        # if '_b' in backbone_arch.lower():
        #     return backbones.EfficientNet(backbone_arch, pretrained, layers_to_freeze+2)
        # else:
        #     return backbones.EfficientNet(model_name='efficientnet_b0',
        #                                   pretrained=pretrained, 
        #                                   layers_to_freeze=layers_to_freeze)
        model = get_pretrained_torchvision_model(backbone_arch)

        for name, child in model.features.named_children():
            logging.debug("Freeze all EfficientNet layers up to n.5")
            if name == str(layers_to_freeze):
                break
            for params in child.parameters():
                params.requires_grad = False
        model = model.features
        return model
    elif 'swin' in backbone_arch.lower():
        return backbones.Swin(model_name='swinv2_base_window12to16_192to256_22kft1k', 
                              pretrained=pretrained, 
                              layers_to_freeze=layers_to_freeze)
    elif 'dino' in backbone_arch.lower():
        return backbones.DINOv2(model_name=backbone_arch, num_trainable_blocks=num_trainable_blocks)
    
    elif 'vgg' in backbone_arch.lower():
        model = get_pretrained_torchvision_model(backbone_arch)

        for name, child in model.features.named_children():
            logging.debug("Freeze all VGG layers up to n.5")
            if name == str(layers_to_freeze):
                break
            for params in child.parameters():
                params.requires_grad = False
        model = model.features
        return model
    
    # if 'dinov2' in backbone_arch.lower():
    #     return backbones.DINOv2(model_name=backbone_arch, **backbone_config)

def get_aggregator(agg_arch='mixvpr', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'cosplace' in agg_arch.lower():
        assert 'in_dim' in agg_config
        assert 'out_dim' in agg_config
        # agg_config_tmp = {key:value for key,value in agg_config.items() if key == 'in_dim' or key == 'out_dim'}
        return aggregators.CosPlace(**agg_config)

    elif 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)
    
    elif 'convap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.ConvAP(**agg_config)
    
    elif 'mixvpr' in agg_arch.lower():
        assert 'in_channels' in agg_config
        assert 'out_channels' in agg_config
        assert 'in_h' in agg_config
        assert 'in_w' in agg_config
        assert 'mix_depth' in agg_config
        return aggregators.MixVPR(**agg_config)
    
    elif 'salad' in agg_arch.lower():
        assert 'num_channels' in agg_config
        assert 'num_clusters' in agg_config
        assert 'cluster_dim' in agg_config
        assert 'token_dim' in agg_config
        return aggregators.SALAD(**agg_config)
    
    elif 'avgpool' in agg_arch.lower():
        return aggregators.AvgPool()
