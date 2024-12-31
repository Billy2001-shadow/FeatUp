import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from featup.util import norm, unnorm, pca, remove_axes
from torchvision.transforms import Compose
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2

from depthAnythingv2jbu import DepthAnythingV2JBU

def plot_feats(image, lr, hr, save_path):
    assert len(image.shape) == len(lr.shape) == len(hr.shape) == 3
    # 使用 PCA 将特征降维并可视化
    [lr_feats_pca, hr_feats_pca], _ = pca([lr.unsqueeze(0), hr.unsqueeze(0)], dim=9)
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    
    # 显示原始图像和上采样前后的特征
    ax[0, 0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[1, 0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[2, 0].imshow(image.permute(1, 2, 0).detach().cpu())

    ax[0, 0].set_title("Image", fontsize=22)
    ax[0, 1].set_title("Original", fontsize=22)
    ax[0, 2].set_title("Upsampled Features", fontsize=22)

    ax[0, 1].imshow(lr_feats_pca[0, :3].permute(1, 2, 0).detach().cpu())
    ax[0, 0].set_ylabel("PCA Components 1-3", fontsize=22)
    ax[0, 2].imshow(hr_feats_pca[0, :3].permute(1, 2, 0).detach().cpu())

    ax[1, 1].imshow(lr_feats_pca[0, 3:6].permute(1, 2, 0).detach().cpu())
    ax[1, 0].set_ylabel("PCA Components 4-6", fontsize=22)
    ax[1, 2].imshow(hr_feats_pca[0, 3:6].permute(1, 2, 0).detach().cpu())

    ax[2, 1].imshow(lr_feats_pca[0, 6:9].permute(1, 2, 0).detach().cpu())
    ax[2, 0].set_ylabel("PCA Components 7-9", fontsize=22)
    ax[2, 2].imshow(hr_feats_pca[0, 6:9].permute(1, 2, 0).detach().cpu())

    remove_axes(ax)
    plt.tight_layout()
    
    # 保存图片到指定路径
    plt.savefig(save_path)  
    plt.close(fig)  # 关闭图像，避免内存溢出


# 加载模型
backbone_weight_path = "/home/chenwu/FeatUp/checkpoints/depth_anything_v2_vits.pth" 
model = DepthAnythingV2JBU('vits',use_norm=False)
# 加载模型权重参数
backbone_state_dict = torch.load(backbone_weight_path,map_location='cuda', weights_only=True)
backbone_state_dict = {k: v for k, v in backbone_state_dict.items() if 'pretrained' in k}
missing_keys, unexpected_keys = model.backnone.load_state_dict(backbone_state_dict, strict=False)


upsampler_weight_path = "/home/chenwu/FeatUp/checkpoints/dinov2_jbu_stack_cocostuff_adjust.ckpt"
upsampler_state_dict = torch.load(upsampler_weight_path,map_location='cuda', weights_only=True)["state_dict"]
upsampler_state_dict = {k: v for k, v in upsampler_state_dict.items() if "scale_net" not in k and "downsampler" not in k}
missing_keys, unexpected_keys = model.upsampler.load_state_dict(upsampler_state_dict, strict=False)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")
backbone_params = sum(p.numel() for p in model.backnone.parameters() if p.requires_grad)
print(f"Total number of trainable parameters in backbone: {backbone_params}")
upsampler_params = sum(p.numel() for p in model.upsampler.parameters() if p.requires_grad)
print(f"Total number of trainable parameters in upsampler: {upsampler_params}")

# 加载并处理图像
input_size = 518
# transform = Compose([
#     Resize(
#         width=input_size,
#         height=input_size,
#         resize_target=False,
#         keep_aspect_ratio=False, # True
#         ensure_multiple_of=14,
#         resize_method='lower_bound',
#         image_interpolation_method=cv2.INTER_CUBIC,
#     ),
#     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     PrepareForNet(),
# ])

# 输入预处理有问题
transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop((input_size, input_size)),
    T.ToTensor(),
    norm
])

image_path = "/tmp/sample_images/skate.jpg"
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
# image = transform({'image': image})['image']
# image_tensor = torch.from_numpy(image).unsqueeze(0).cuda()
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).cuda() # torch.Size([1, 3, 224, 224])


upsampler = model.cuda()
lr_feats,hr_feats = upsampler(image_tensor) # torch.Size([1, 384, 37, 37]) torch.Size([1, 384, 592, 592]) 16倍放缩 
upsampler.cpu()

# 指定保存的路径
save_path = "test_results/test1.png"

# 可视化结果并保存图像
plot_feats(unnorm(image_tensor)[0], lr_feats[0], hr_feats[0], save_path)

print(f"Image saved to {save_path}")