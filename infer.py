import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from featup.util import norm, unnorm, pca, remove_axes
import requests

# from net import dinov2

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


# 只下载一个测试图像（例如：skate.jpg）
image_path = "/tmp/sample_images/skate.jpg"
# 选择模型为 vit
model_option = 'dinov2'
model = torch.hub.load("mhamilton723/FeatUp", model_option, use_norm=False)
# model_pth = "checkpoints/dinov2_jbu_stack_cocostuff.ckpt"
# model = torch.hub.load(model_pth,model_option)
# model = torch.hub.load("./",model_option,source='local')
# weight_pth = "checkpoints/dinov2_jbu_stack_cocostuff.ckpt"
# model = dinov2(weight_path=weight_pth, use_norm=False)


# 加载并处理图像
input_size = 224
transform = T.Compose([
    T.Resize(input_size),
    T.CenterCrop((input_size, input_size)),
    T.ToTensor(),
    norm
])

image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).cuda() # torch.Size([1, 3, 224, 224])

# 使用模型进行特征上采样
upsampler = model.cuda()
hr_feats = upsampler(image_tensor)                  # torch.Size([1, 384, 256, 256])    高分辨率特征图
lr_feats = upsampler.model(image_tensor)            # torch.Size([1, 384, 16, 16])      低分辨率特征图
upsampler.cpu()

# 指定保存的路径
save_path = "./upsampled_feats.png"

# 可视化结果并保存图像
plot_feats(unnorm(image_tensor)[0], lr_feats[0], hr_feats[0], save_path)

print(f"Image saved to {save_path}")
