import torch
from featup.layers import ChannelNorm
from featup.upsamplers import get_upsampler
from torch.nn import Module

from depth_anything_v2.dpt import DepthAnythingV2

dependencies = ['torch', 'torchvision', 'PIL', 'featup']  # List any dependencies here

# 定义网络结构
class DepthAnythingV2JBU(Module):

    def __init__(self, encoder='vits',use_norm=False):
        super().__init__()
        # 这里把TinyVit加载进来
        self.use_norm = use_norm
        self.encoder = encoder
        self.embed_dim = 384
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        backnone = DepthAnythingV2(**model_configs[self.encoder])
        
        if self.use_norm:
            self.backnone = torch.nn.Sequential(backnone, ChannelNorm(self.embed_dim))
        else:
            self.backnone = backnone
        self.upsampler = get_upsampler("jbu_stack", self.embed_dim)
       

    def forward(self, image):
        features = self.backnone(image)  #torch.Size([1, 1369, 384]) 
        h = w = int(features[-1].shape[1] ** 0.5)
        embed_dim = self.embed_dim
        feature_map = features[-1].view(1,h,w,embed_dim).permute(0, 3, 1, 2) # torch.Size([1, 384, 37, 37])

        return feature_map,self.upsampler(feature_map, image)

  







