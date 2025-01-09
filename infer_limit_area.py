import os

from tqdm import tqdm
from PIL import Image
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu, load_checkpoint

from networks import U2NET


import os
from PIL import Image
from send2trash import send2trash

def move_all_to_trash(directory):
    # 遍历目录中的所有文件和子目录
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # 使用 send2trash 将文件或目录移到回收站
        try:
            send2trash(file_path)
            print(f"已将 '{file_path}' 移动到回收站")
        except Exception as e:
            print(f"移动 '{file_path}' 到回收站时出错: {e}")

def resize_images_in_directory(directory, out_dir, target_area):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 获取文件的完整路径
        file_path = os.path.join(directory, filename)
        
        # 检查文件是否为图片文件（通过扩展名）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                # 打开图像
                img = Image.open(file_path)
                
                # 获取原始的宽度和高度
                width, height = img.size
                
                # 计算当前图像的面积
                current_area = width * height
                
                # 如果当前图像面积大于目标面积，进行缩放
                if current_area > target_area:
                    # 计算目标宽度和高度，保持长宽比例
                    ratio = (target_area / current_area) ** 0.5
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    
                    # 缩放图像
                    resized_img = img.resize((new_width, new_height))
                    
                    # 直接覆盖原始图像
                    resized_img.save(os.path.join(out_dir, filename))
                    print(f"已覆盖原始图像：{file_path}")
                else:
                    print(f"图像 {filename} 的面积已经小于目标面积，不做处理。")
            except Exception as e:
                print(f"处理图像 {filename} 时出错：{e}")
        else:
            print(f"跳过非图片文件：{filename}")


device = "cuda"

image_resize_dir = "input_images_resize"
result_dir = "output_images"
checkpoint_path = os.path.join("trained_checkpoint", "checkpoint_u2net.pth")
do_palette = True

src_input_images_dir = "input_images"
target_area = 500*500
move_all_to_trash(image_resize_dir)
resize_images_in_directory(src_input_images_dir, image_resize_dir, target_area)


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint(net, checkpoint_path)
net = net.to(device)
net = net.eval()

palette = get_palette(4)

images_list = sorted(os.listdir(image_resize_dir))
pbar = tqdm(total=len(images_list))
for image_name in images_list:
    img = Image.open(os.path.join(image_resize_dir, image_name)).convert("RGB")
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
    if do_palette:
        output_img.putpalette(palette)
    output_img.save(os.path.join(result_dir, image_name[:-3] + "png"))

    pbar.update(1)

pbar.close()


