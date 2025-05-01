from dataclasses import dataclass
from .utils import check_image_type
import depth_pro
import cv2
import matplotlib
import numpy as np
import torch
import os
from .download_weights import *
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

from depth_pro.network.vit_factory import ViTPreset

@dataclass
class DepthProConfig:
    """Configuration for DepthPro."""

    patch_encoder_preset: ViTPreset
    image_encoder_preset: ViTPreset
    decoder_features: int

    checkpoint_uri: Optional[str] = None
    fov_encoder_preset: Optional[ViTPreset] = None
    use_fov_head: bool = True

class Depth_Pro:
    def __init__(self):
        DEFAULT_MONODEPTH_CONFIG_DICT = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri="./weights/depth_pro.pt",
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        
        weight_path = "weights/depth_pro.pt"
        if not os.path.exists(weight_path):
            download_weights_depthpro(weight_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = depth_pro.create_model_and_transforms(config=DEFAULT_MONODEPTH_CONFIG_DICT, device=self.device)
        self.model.eval()
    
    def __call__(self, image):
        return self.run(image)

    def run(self, image):
        image = check_image_type(image)
        image = self.transform(image)
        prediction = self.model.infer(image)
        depth = prediction["depth"]
        depth_cpu = depth.to("cpu").detach().numpy().copy()
        depth_image = self.__render_depth(1/depth_cpu)
        return {"image": depth_image, "depth": depth_cpu}

    def __render_depth(self, values, colormap_name="magma_r"):
        min_value, max_value = values.min(), values.max()
        normalized_values = (values - min_value) / (max_value - min_value)

        colormap = matplotlib.colormaps[colormap_name]
        colors = colormap(normalized_values, bytes=True)  # ((1)xhxwx4)
        colors = colors[:, :, :3]  # Discard alpha component
        return cv2.cvtColor(np.array(colors), cv2.COLOR_BGR2RGB)
