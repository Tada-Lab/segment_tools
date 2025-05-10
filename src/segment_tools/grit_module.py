import sys
import torch
import os

from detectron2.config import get_cfg

sys.path.append(os.path.join(os.path.dirname(__file__), "GRiT"))
sys.path.append(os.path.join(os.path.dirname(__file__), "GRiT/third_party/CenterNet2/projects/CenterNet2/"))
from centernet.config import add_centernet_config
from grit.config import add_grit_config

from grit.predictor import VisualizationDemo

from .download_weights import *

from .utils import mask_class_objects, draw_multi_mask, check_image_type, mask_class_objects_multi

# GRiTの情報(cap)を取得
# from GRiT.demo_process import GRiT_process
# from detectron2.data.detection_utils import _apply_exif_orientation

def setup_cfg(test_task="", weight_path="GRiT/models/grit_b_densecap_objectdet.pth"):
    cfg = get_cfg()
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(os.path.join(os.path.dirname(__file__), "GRiT/configs/GRiT_B_DenseCap_ObjectDet.yaml"))
    opts = ["MODEL.WEIGHTS", weight_path]
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    if test_task:
        cfg.MODEL.TEST_TASK = test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg

class GRiT:
    def __init__(self):
        weight_path = "weights/grit_b_densecap_objectdet.pth"
        if not os.path.exists(weight_path):
            download_weights_grit(weight_path)

        self.cfg = setup_cfg(weight_path=weight_path)
        self.demo = VisualizationDemo(self.cfg)
        
    def __call__(self, image):
        return self.run(image)

    def run(self, image):
        image = check_image_type(image, "numpy")
        predictions, visualized_output = self.demo.run_on_image(image)

        # 中身をcpuに移動
        instances = predictions["instances"].to(torch.device("cpu"))
        # 検出したbox
        pred_boxes = instances.pred_boxes.tensor.detach().numpy()
        # それへのキャプショニング
        pred_object_descriptions = instances.pred_object_descriptions.data
    
        output_image = visualized_output.get_image()[:, :, ::-1]
        return {"image": output_image, "bbox": pred_boxes, "info": pred_object_descriptions}