import os

# defalut weight paths
package_dir = os.path.dirname(os.path.abspath(__file__))

# if nvidia gpu is available, use pytorch to inference, else use tinygrad
try:
    import torch
    if torch.cuda.is_available():
        from .segnet_torch import SegNet
        DEFAULT_CKPT_PATH = os.path.join(package_dir,'universal_tiny.pth')
    else:
        raise ImportError
except ImportError:
    from .segnet_tinygrad import SegNet
    DEFAULT_CKPT_PATH = os.path.join(package_dir,'universal_tiny.safetensors')