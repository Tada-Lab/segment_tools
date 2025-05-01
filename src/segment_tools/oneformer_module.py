# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
setup_logger(name="oneformer")

# Import libraries
import numpy as np
import cv2
import torch

# from google.colab.patches import cv2_imshow
import imutils

# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog

import os
import sys
import PIL

sys.path.append(os.path.join(os.path.dirname(__file__), "OneFormer_segtools"))
from demo.defaults import DefaultPredictor
from demo.visualizer import Visualizer, ColorMode


# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

from .download_weights import *

from .utils import mask_class_objects, draw_multi_mask, check_image_type, mask_class_objects_multi

from itertools import cycle

cpu_device = torch.device("cpu")
config_dir = os.path.join(os.path.dirname(__file__), "OneFormer_segtools/configs")
SWIN_CFG_DICT = {
    "ade20k":       "ade20k/swin/oneformer_swin_large_bs16_160k.yaml",
    "cityscapes":   "cityscapes/swin/oneformer_swin_large_bs16_90k.yaml",
    "coco":         "coco/swin/oneformer_swin_large_bs16_100ep.yaml",
    "vistas":       "mapillary_vistas/swin/oneformer_swin_large_bs16_300k.yaml", 
}

CONVNEXT_CFG_DICT = {
    "ade20k":       "ade20k/convnext/oneformer_convnext_large_bs16_160k.yaml",
    "cityscapes":   "cityscapes/convnext/oneformer_convnext_large_bs16_90k.yaml",
    "coco":         "",
    "vistas":       "mapillary_vistas/convnext/oneformer_convnext_large_bs16_300k.yaml",
}

DINAT_CFG_DICT = {
    "ade20k":       "ade20k/dinat/coco_pretrain_oneformer_dinat_large_bs16_160k_1280x1280.yaml",
    "cityscapes":   "cityscapes/dinat/oneformer_dinat_large_bs16_90k.yaml",
    "coco":         "coco/dinat/oneformer_dinat_large_bs16_100ep.yaml",
    "vistas":       "mapillary_vistas/dinat/oneformer_dinat_large_bs16_300k.yaml",
}

SWIN_CFG_DICT = {k: os.path.join(config_dir, v) for k, v in SWIN_CFG_DICT.items()}
CONVNEXT_CFG_DICT = {k: os.path.join(config_dir, v) for k, v in CONVNEXT_CFG_DICT.items()}
DINAT_CFG_DICT = {k: os.path.join(config_dir, v) for k, v in DINAT_CFG_DICT.items()}

cfg_backbone_dict = {
    "swin": SWIN_CFG_DICT,
    "convnext": CONVNEXT_CFG_DICT,
    "dinat": DINAT_CFG_DICT,
}

def setup_cfg(dataset, model_path, backbone):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    
    cfg_path = cfg_backbone_dict[backbone][dataset]
    
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg


def setup_modules(dataset, model_path, backbone):
    cfg = setup_cfg(dataset, model_path, backbone)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )
    if "cityscapes_fine_sem_seg_val" in cfg.DATASETS.TEST_PANOPTIC[0]:
        from cityscapesscripts.helpers.labels import labels

        stuff_colors = [k.color for k in labels if k.trainId != 255]
        metadata = metadata.set(stuff_colors=stuff_colors)

    return predictor, metadata


def panoptic_run(img, predictor, metadata, no_image=False):
    
    if not no_image:
        # img:numpy img_にcopy
        img_ = np.copy(img)
        visualizer = Visualizer(
            img_[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE
        )
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    
    if not no_image:
        out = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg.to(cpu_device), segments_info, alpha=0.5
        )
    else:
        out = None
    classes = metadata.stuff_classes

    # クラス名も追加
    for idx, segment_info in enumerate(segments_info):
        segments_info[idx].update({"class": classes[segment_info["category_id"]]})
        
    panoptic_seg = panoptic_seg.cpu().numpy()
    return out, panoptic_seg, segments_info


def instance_run(img, predictor, metadata):
    visualizer = Visualizer(
        img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE
    )
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to(cpu_device)
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    return out


def semantic_run(img, predictor, metadata):
    visualizer = Visualizer(
        img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE
    )
    predictions = predictor(img, "semantic")
    out = visualizer.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=0.5
    )
    return out


TASK_INFER = {
    "panoptic": panoptic_run,
    "instance": instance_run,
    "semantic": semantic_run,
}

ade20k_weight_dict = {
    "swin": "250_16_swin_l_oneformer_ade20k_160k.pth",
    "convnext": "250_16_convnext_l_oneformer_ade20k_160k.pth",
    "dinat": "250_16_dinat_l_oneformer_ade20k_160k.pth",
}

cityscapes_weight_dict = {
    "swin": "250_16_swin_l_oneformer_cityscapes_90k.pth",
    "convnext": "250_16_convnext_l_oneformer_cityscapes_90k.pth",
    "dinat": "250_16_dinat_l_oneformer_cityscapes_90k.pth",
}

coco_weight_dict = {
    "swin": "150_16_swin_l_oneformer_coco_100ep.pth",
    "convnext": "", # not supported
    "dinat": "150_16_dinat_l_oneformer_coco_100ep.pth",
}

vistas_weight_dict = {
    "swin": "250_16_swin_l_oneformer_mapillary_300k.pth",
    "convnext": "250_16_convnext_l_oneformer_mapillary_300k.pth",
    "dinat": "250_16_dinat_l_oneformer_mapillary_300k.pth",
}

class OneFormer:
    def __init__(self, dataset="ade20k", backbone="dinat", weight_dir="weights"):
        if dataset == "ade20k":
            weight_path = ade20k_weight_dict[backbone]
            weight_path = os.path.join(weight_dir, weight_path)
            
            if not os.path.exists(weight_path):
                download_weights_ade20k(weight_path, backbone)
            self.predictor, self.metadata = setup_modules(dataset, weight_path, backbone)
            
        elif dataset == "cityscapes":
            weight_path = cityscapes_weight_dict[backbone]
            weight_path = os.path.join(weight_dir, weight_path)
            
            if not os.path.exists(weight_path):
                download_weights_cityscapes(weight_path, backbone)
            self.predictor, self.metadata = setup_modules(dataset, weight_path, backbone)
            
        elif dataset == "coco":
            weight_path = coco_weight_dict[backbone]
            if weight_path == "": raise ValueError("convnext is not supported in coco dataset")
            weight_path = os.path.join(weight_dir, weight_path)
            
            if not os.path.exists(weight_path):
                download_weights_coco(weight_path, backbone)
            self.predictor, self.metadata = setup_modules(dataset, weight_path, backbone)
            
        elif dataset == "vistas":
            weight_path = vistas_weight_dict[backbone]
            weight_path = os.path.join(weight_dir, weight_path)
            
            if not os.path.exists(weight_path):
                download_weights_vistas(weight_path, backbone)
            self.predictor, self.metadata = setup_modules(dataset, weight_path, backbone)
                
        else:
            raise ValueError("dataset is not supported")
        
    def __call__(self, image, prompt=None, color="random", alpha=0.5, task="panoptic", panoptic_mask=False, no_image=False):
        return self.run(image, prompt, color, alpha, task, panoptic_mask, no_image)
                
    def run(self, image, prompt=None, color="random", alpha=0.5, task="panoptic", panoptic_mask=False, no_image=False):
        image = check_image_type(image)
        # もしimageの次元が3以上(RGBAなど)の場合、RGBに変換
        if image.shape[-1] > 3:
            image = image[:, :, :3]
        out, panoptic_seg, segments_info = TASK_INFER[task](image, self.predictor, self.metadata, no_image=no_image)

        try:
            out = out[0] if len(out) == 2 else out
        except:
            pass

        # promptがNoneでない場合のみ、draw_multi_maskを実行(多分重いので)
        # panoptic_maskがTrueの場合、colorはrandomしかできない(色指定する方も面倒かも)
        if prompt is not None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            color = [color] if isinstance(color, str) else color
            prompt_color_map = {prompt_: color_ for prompt_, color_ in zip(prompt, cycle(color))}
            panoptic_seg, output_image = mask_class_objects_multi(seg=panoptic_seg, ann=segments_info, stuff_classes=self.metadata.stuff_classes, image=image, panoptic_mask=panoptic_mask, prompt_color_map=prompt_color_map, alpha=alpha)
        else:
            try:
                output_image = out.get_image()[:, :, ::-1]
            except:
                output_image = None
        
        return {"image": output_image, "mask": panoptic_seg, "info": segments_info}
                
    def get_labels(self):
        return self.metadata.stuff_classes
