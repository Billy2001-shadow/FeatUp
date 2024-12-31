import torch
from featup.layers import ChannelNorm
from featup.upsamplers import get_upsampler
from torch.nn import Module

from TinyVit.tinyvit.tiny_vit import TinyViT

dependencies = ['torch', 'torchvision', 'PIL', 'featup']  # List any dependencies here

# 定义网络结构
class TinyVitJBU(Module):

    def __init__(self, weight_path, use_norm):
        super().__init__()
        # 这里把TinyVit加载进来
        self.input_size = (224,224)
        self.embed_dims = [64, 128, 160, 320]
        self.num_heads = [2, 4, 5, 10]
        self.window_sizes = [7, 7, 14, 7]
        self.drop_path_rate = 0.0
        model = TinyViT(img_size=self.input_size,
                        embed_dims=self.embed_dims,
                        num_heads=self.num_heads,
                        window_sizes=self.window_sizes,
                        drop_path_rate=self.drop_path_rate,)

        # 只加载pretrained部分参数
        state_dict = torch.load(weight_path,map_location='cuda', weights_only=True)
        if 'model' in state_dict:
            state_dict = state_dict['model']
            filtered_state_dict = {k: v for k, v in state_dict.items() if 'pretrained' in k}
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        
        if use_norm:
            self.model = torch.nn.Sequential(model, ChannelNorm(self.embed_dims[-1]))
        else:
            self.model = model
        self.upsampler = get_upsampler("jbu_stack", self.embed_dims[-1])

    def forward(self, image):
        features = self.model(image)  # torch.Size([1, 784, 128])  torch.Size([1, 196, 160]) torch.Size([1, 49, 320]) torch.Size([1, 49, 320])
        
        h = w = int(features[-1].shape[1] ** 0.5)
        embed_dim = features[-1].shape[2]
        feature_map = features[-1].view(1,h,w,embed_dim).permute(0, 3, 1, 2)
        
        return feature_map,self.upsampler(feature_map, image)
        







