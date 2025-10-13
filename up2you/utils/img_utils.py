import numpy as np
import random
from PIL import Image
import torch
from typing import Union
from torchvision.utils import save_image

def get_bg_color(bg_color):
    if bg_color == "white":
        bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    elif bg_color == "black":
        bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    elif bg_color == "gray":
        bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    elif bg_color == "random":
        bg_color = np.random.rand(3)
    elif bg_color == "random_gray":
        bg_color = random.uniform(0.3, 0.7)
        bg_color = np.array([bg_color] * 3, dtype=np.float32)
    elif isinstance(bg_color, float):
        bg_color = np.array([bg_color] * 3, dtype=np.float32)
    elif isinstance(bg_color, list) or isinstance(bg_color, tuple):
        bg_color = np.array(bg_color, dtype=np.float32)
    else:
        raise NotImplementedError
    return bg_color

def load_image(
    image: Union[str, Image.Image],
    height: int,
    width: int,
    background_color: torch.Tensor = torch.tensor([0.5, 0.5, 0.5]),
    rescale: bool = False,
    mask_aug: bool = False,
    return_alpha: bool = False,
):
    if isinstance(image, str):
        image = Image.open(image)

    image = image.resize((width, height))
    image = torch.from_numpy(np.array(image)).float() / 255.0

    alpha = image[:, :, 3]  # Extract alpha channel
    if mask_aug:
        h, w = alpha.shape
        y_indices, x_indices = torch.where(alpha > 0.5)
        if len(y_indices) > 0 and len(x_indices) > 0:
            idx = torch.randint(len(y_indices), (1,)).item()
            y_center = y_indices[idx].item()
            x_center = x_indices[idx].item()
            mask_h = random.randint(h // 8, h // 4)
            mask_w = random.randint(w // 8, w // 4)

            y1 = max(0, y_center - mask_h // 2)
            y2 = min(h, y_center + mask_h // 2)
            x1 = max(0, x_center - mask_w // 2)
            x2 = min(w, x_center + mask_w // 2)

            alpha[y1:y2, x1:x2] = 0.0
            image[:, :, 3] = alpha

    image = image[:, :, :3] * image[:, :, 3:4] + background_color * (
        1 - image[:, :, 3:4]
    )
    if rescale:
        image = image * 2.0 - 1.0
    if return_alpha:
        return image, alpha
    return image # H W C -> 1 C H W torchvision.utils save_image(image, "image.png")

def concat_images_horizontally(image_list, save_path):
    """
    将图像列表横向拼接并保存
    
    Args:
        image_list: PIL Image对象列表
        save_path: 保存路径
    """
    if not image_list:
        return
    
    # 获取图像尺寸
    widths, heights = zip(*(img.size for img in image_list))
    total_width = sum(widths)
    max_height = max(heights)
    
    # 创建新图像
    concatenated = Image.new('RGB', (total_width, max_height))
    
    # 拼接图像
    x_offset = 0
    for img in image_list:
        concatenated.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    concatenated.save(save_path)

def preprocess_image(image: Image.Image, height, width):
    image = np.array(image)
    alpha = image[..., 3] > 0
    H, W = alpha.shape
    # get the bounding box of alpha
    y, x = np.where(alpha)
    y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
    x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
    image_center = image[y0:y1, x0:x1]
    # resize the longer side to H * 0.9
    H, W, _ = image_center.shape
    if H > W:
        W = int(W * (height * 0.9) / H)
        H = int(height * 0.9)
    else:
        H = int(H * (width * 0.9) / W)
        W = int(width * 0.9)
    image_center = np.array(Image.fromarray(image_center).resize((W, H)))
    # pad to H, W
    start_h = (height - H) // 2
    start_w = (width - W) // 2
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[start_h : start_h + H, start_w : start_w + W] = image_center
    image = image.astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    return image

def process_image_rgba(image_rgba, res=1024, ratio=0.8, background_color=[0.5,0.5,0.5]):
    """
    Process image to fit the canvas, handling non-square images properly

    Args:
        image_rgba: PIL Image, RGBA format
        res: canvas size [res, res]
        ratio: ratio of foreground in canvas
        background_color: background color [R, G, B], range [0,1]
    
    Returns:
        PIL Image: processed RGBA image
    """

    if image_rgba.mode != 'RGBA':
        image_rgba = image_rgba.convert('RGBA')
    
    alpha_channel = image_rgba.split()[-1]
    
    bbox = alpha_channel.getbbox()
    if bbox is None:
        canvas = Image.new('RGBA', (res, res), (0, 0, 0, 0))
        return canvas
    
    cropped_image = image_rgba.crop(bbox)
    cropped_width, cropped_height = cropped_image.size
    
    # Calculate target size maintaining aspect ratio
    target_size = int(res * ratio)
    if cropped_width > cropped_height:
        new_width = target_size
        new_height = int(cropped_height * target_size / cropped_width)
    else:
        new_height = target_size
        new_width = int(cropped_width * target_size / cropped_height)
    
    resized_image = cropped_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    canvas = Image.new('RGBA', (res, res), (0, 0, 0, 0))
    
    orig_width, orig_height = image_rgba.size
    
    # 计算前景区域在原始图像中的中心位置
    orig_center_x = (bbox[0] + bbox[2]) / 2
    orig_center_y = (bbox[1] + bbox[3]) / 2
    
    # 处理非正方形图像：将原始图像映射到正方形空间
    # 找到原始图像的最大边长，用于创建虚拟正方形空间
    max_orig_dim = max(orig_width, orig_height)
    
    # 计算原始图像在虚拟正方形空间中的偏移
    offset_x = (max_orig_dim - orig_width) / 2
    offset_y = (max_orig_dim - orig_height) / 2
    
    # 将原始中心位置映射到虚拟正方形空间
    virtual_center_x = orig_center_x + offset_x
    virtual_center_y = orig_center_y + offset_y
    
    # 计算在目标画布上的相对位置
    rel_center_x = virtual_center_x / max_orig_dim
    rel_center_y = virtual_center_y / max_orig_dim
    
    # 计算在目标画布上的绝对位置
    canvas_center_x = int(rel_center_x * res)
    canvas_center_y = int(rel_center_y * res)
    
    # 计算粘贴位置
    paste_x = canvas_center_x - new_width // 2
    paste_y = canvas_center_y - new_height // 2
    
    # 确保粘贴位置在画布范围内
    paste_x = max(0, min(paste_x, res - new_width))
    paste_y = max(0, min(paste_y, res - new_height))
    
    canvas.paste(resized_image, (paste_x, paste_y), resized_image)
    
    bg_color_255 = tuple(int(c * 255) for c in background_color)
    
    canvas_alpha = canvas.split()[-1]
    
    canvas_rgb = canvas.convert('RGB')
    bg_canvas_rgb = Image.new('RGB', (res, res), bg_color_255)
    
    final_rgb = Image.composite(canvas_rgb, bg_canvas_rgb, canvas_alpha)
    final_rgba = Image.merge('RGBA', (*final_rgb.split(), canvas_alpha))
    
    return final_rgba