# hubconf.py
import torch
from featup.featurizers.util import get_featurizer
from featup.layers import ChannelNorm
from featup.upsamplers import get_upsampler
from torch.nn import Module
import os


dependencies = ['torch', 'torchvision', 'PIL', 'featup']  # List any dependencies here


class UpsampledBackbone(Module):

    def __init__(self, model_name, use_norm):
        super().__init__()
        model, patch_size, self.dim = get_featurizer(model_name, "token", num_classes=1000)
        if use_norm:
            self.model = torch.nn.Sequential(model, ChannelNorm(self.dim))
        else:
            self.model = model
        self.upsampler = get_upsampler("jbu_stack", self.dim) # upsammpler网络结构

    def forward(self, image):
        return self.upsampler(self.model(image), image)


def _load_backbone(weight_path, use_norm, model_name):
    """
    The function that will be called by Torch Hub users to instantiate your model.
    Args:
        pretrained (bool): If True, returns a model pre-loaded with weights.
    Returns:
        An instance of your model.
    """
    model = UpsampledBackbone(model_name, use_norm) # model_name=dinov2 use_norm=False
    
    if weight_path:
        # 如果提供了本地模型路径，从本地加载权重
        if os.path.exists(weight_path):
            print(f"Loading model weights from local file: {weight_path}")
            state_dict = torch.load(weight_path)["state_dict"]
        else:
            raise FileNotFoundError(f"Local model file not found at {weight_path}")

        # 过滤状态字典，去除不需要的部分
        state_dict = {k: v for k, v in state_dict.items() if "scale_net" not in k and "downsampler" not in k}
        model.load_state_dict(state_dict, strict=False)

    return model


def vit(pretrained=True, use_norm=True):
    return _load_backbone(pretrained, use_norm, "vit")


def dino16(pretrained=True, use_norm=True):
    return _load_backbone(pretrained, use_norm, "dino16")


def clip(pretrained=True, use_norm=True):
    return _load_backbone(pretrained, use_norm, "clip")


def dinov2(weight_path, use_norm=True):
    return _load_backbone(weight_path, use_norm, "dinov2")
# def dinov2(pretrained=True, use_norm=True):
#     return _load_backbone(pretrained, use_norm, "dinov2")


def resnet50(pretrained=True, use_norm=True):
    return _load_backbone(pretrained, use_norm, "resnet50")

def maskclip(pretrained=True, use_norm=True):
    assert not use_norm, "MaskCLIP only supports unnormed model"
    return _load_backbone(pretrained, use_norm, "maskclip")
