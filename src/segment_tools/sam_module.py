import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import matplotlib.pyplot as plt
import os

vit_list = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
from .utils import mask_class_objects, draw_multi_mask, check_image_type, mask_class_objects_multi

class SAM:
    def __init__(self, vit_size="vit_h"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weight_path = os.path.join("weights", vit_list[vit_size])

        if not os.path.exists(os.path.join("weights", vit_list[vit_size])):
            from .download_weights import download_weights_SAM
            download_weights_SAM(weight_path, vit_size)

        self.sam = sam_model_registry[vit_size](checkpoint=weight_path).to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.predictor = SamPredictor(self.sam)
    
    def run(self, image, input_points=None, input_label=None, multimask_output=False):
        image = check_image_type(image)
        
        if input_points is None:
            masks = self.mask_generator.generate(image)
            segmentation_arrays = [mask["segmentation"] for mask in masks]
            result = np.stack(segmentation_arrays)
            print(result.shape)
            output_image = draw_multi_mask(result, image, panoptic_mask=True)
        else:
            self.predictor.set_image(image)

            if isinstance(input_points, list):
                input_points = np.array(input_points)
            if input_label is None:
                num_points = input_points.shape[0]
                input_label = np.ones(num_points, dtype=int)

            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_label,
                multimask_output=multimask_output,
            )

            output_image = draw_multi_mask(masks, image)

        return {"image": output_image, "ann": masks}