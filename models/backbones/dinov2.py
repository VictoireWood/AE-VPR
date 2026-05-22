import torch
import torch.nn as nn
import numpy as np

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

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
        ):
        super().__init__()  # NOTE - 调用nn.Module的初始化方法

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        # self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = torch.hub.load('/root/.cache/torch/hub/facebookresearch_dinov2_main', model_name, trust_repo=True, source='local')
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token


    def forward(self, x: torch.Tensor):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        # SECTION
        B, C, H, W = x.shape

        if H % 14 != 0 or W % 14 != 0:
            target_size = ((H // 14) * 14, (W // 14) * 14)
            size = (H, W)
            cut_size_l = ((size[0]-target_size[0])//2, (size[1]-target_size[1])//2)
            size_r = (target_size[0] + cut_size_l[0], target_size[1] + cut_size_l[1])
            x = x[:,:, cut_size_l[0]:size_r[0], cut_size_l[1]:size_r[1]]
        # !SECTION

        half = x.dtype == torch.float16
        if half:
            x = x.to(torch.float)
        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x) # NOTE - 这里是不是调用父类的方法？或者是DINOv2自己的方法？(应该是dinov2的自带方法)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if half:
            f = f.to(torch.float16)
            t = t.to(torch.float16)


        if self.return_token:
            return f, t
        return f


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')

if __name__ == '__main__':

    size = (360, 480)
    target_size = ((size[0]//14)*14, (size[1]//14)*14)
    cut_size_l = ((size[0]-target_size[0])//2, (size[1]-target_size[1])//2)
    cut_size_r = (size[0] - cut_size_l[0] - target_size[0], size[1] - cut_size_l[1] - target_size[1])
    size_r = (target_size[0] + cut_size_l[0], target_size[1] + cut_size_l[1])
    x = torch.randn(13, 3, size[0], size[1])
    x = x[:,:, cut_size_l[0]:size_r[0], cut_size_l[1]:size_r[1]]
    m = DINOv2(model_name='dinov2_vitb14',
                        # pretrained=True,
                        # layers_to_freeze=7,
                        # layers_to_crop = [],
                        num_trainable_blocks = 2,
                        )
    r = m(x)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')