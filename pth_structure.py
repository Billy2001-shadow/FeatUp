import torch

def analyze_pth_file(pth_file_path):
    """
    分析 .pth 文件内容，打印保存的键及其数据类型和形状。
    
    Parameters:
        pth_file_path (str): .pth 文件路径
    """
    try:
        # 加载 .pth 文件
        data = torch.load(pth_file_path, map_location="cpu")
        
        # 判断是否是字典格式
        if isinstance(data, dict):
            print(f"Loaded .pth file contains a dictionary with {len(data)} keys:")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"Key: {key}, Type: Tensor, Shape: {value.shape}")
                elif isinstance(value, (dict, list)):
                    print(f"Key: {key}, Type: {type(value).__name__}, Length: {len(value)}")
                else:
                    print(f"Key: {key}, Type: {type(value).__name__}, Value: {value}")
        elif isinstance(data, list):
            print(f"Loaded .pth file contains a list with {len(data)} elements:")
            for idx, value in enumerate(data):
                print(f"Index: {idx}, Type: {type(value).__name__}")
        else:
            print(f"Loaded .pth file contains data of type: {type(data).__name__}")
    except Exception as e:
        print(f"Failed to analyze .pth file: {e}")


def print_dict():
    upsampler_weight_path = "checkpoints/dinov2_jbu_stack_cocostuff_adjust.ckpt"
    upsampler_state_dict = torch.load(upsampler_weight_path,map_location='cuda', weights_only=True)["state_dict"]
    upsampler_state_dict = {k: v for k, v in upsampler_state_dict.items() if "scale_net" not in k and "downsampler" not in k}
    for key, value in upsampler_state_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"Key: {key}, Type: Tensor, Shape: {value.shape}")
        elif isinstance(value, (dict, list)):
            print(f"Key: {key}, Type: {type(value).__name__}, Length: {len(value)}")
        else:
            print(f"Key: {key}, Type: {type(value).__name__}, Value: {value}")

if __name__ == "__main__":
    # 替换为你的 .pth 文件路径
    pth_file_path = "checkpoints/latest_epoch_19.pth"
    # analyze_pth_file(pth_file_path)
    print_dict()

 
# tiny_vit_5m_22kto1k_distill.pth 只有一个键：model  Length: 296  
# latest_epoch_19.pth 有三个键：model( Length: 442), optimizer, epoch
# depth_anything_v2_vits.pth 有239个键，直接是权重的名称(pretrained+depth_head)
# dinov2_jbu_stack_cocostuff_adjust.ckpt 有8个键：state_dict(Length: 42)  epoch global_step pytorch-lightning_version loops callbacks optimizer_states lr_schedulers