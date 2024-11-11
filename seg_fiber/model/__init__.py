import os
import torch

# defalut weight paths
package_dir = os.path.dirname(os.path.abspath(__file__))

# if nvidia gpu is available, use pytorch to inference, else use tinygrad
if torch.cuda.is_available():
    print('=== PyTorch Model ===')
    from .unet_torch import SegNet
    DEFAULT_CKPT_PATH = os.path.join(package_dir,'universal_tiny.pth')
else:
    print('=== Tinygrad Model ===')
    from .unet_tinygrad import SegNet
    DEFAULT_CKPT_PATH = os.path.join(package_dir,'universal_tiny.safetensors')
