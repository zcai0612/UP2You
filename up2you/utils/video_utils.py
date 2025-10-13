import imageio
import numpy as np


def tensor_to_video(tensor, video_path):
    tensor = tensor.cpu().numpy()
    tensor = tensor.transpose(0, 2, 3, 1)
    tensor = (tensor * 255).astype(np.uint8)
    imageio.mimsave(video_path, tensor)