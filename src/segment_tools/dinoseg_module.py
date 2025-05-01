from PIL import Image

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, predict

# segment anything
from segment_anything import build_sam, SamPredictor, sam_model_registry
import numpy as np

# diffusers
import torch
from huggingface_hub import hf_hub_download
import os
from .utils import draw_multi_mask

sam_pth = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}

class DINOSeg:
    def __init__(
        self,
        ckpt_repo_id="ShilongLiu/GroundingDINO",
        ckpt_filenmae="groundingdino_swinb_cogcoor.pth",
        ckpt_config_filename="GroundingDINO_SwinB.cfg.py",
        sam_checkpoint="vit_h",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if sam_checkpoint not in sam_pth:
            raise ValueError(f"Invalid sam_checkpoint: {sam_checkpoint}")
        sam_checkpoint_path = os.path.join("weights", sam_pth[sam_checkpoint])
        if not os.path.exists(sam_checkpoint_path):
            from .download_weights import download_weights_SAM
            download_weights_SAM(sam_checkpoint_path, sam_checkpoint)
        self.groundingdino_model = self.__load_model_hf(
            ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, self.device
        )
        self.sam_predictor = SamPredictor(
            sam_model_registry[sam_checkpoint](checkpoint=sam_checkpoint_path).to(self.device)
        )
        
    def __call__(self, image, text):
        return self.run(image, text)

    def run(self, image, text):
        """dinoを用いた画像のセグメンテーション

        Args:
            image: PILでもnumpyでもパスでも可
            text: テキストプロンプト

        Returns:
            image: セグメンテーション結果を描画した画像
            mask: セグメンテーション結果のマスク
            bbox: 物体検出結果のバウンディングボックス
        """
        image_source, image = self.__load_image(image)
        annotated_frame, detected_boxes = self.__detect(
            image, image_source, text_prompt=text, model=self.groundingdino_model
        )
        if len(detected_boxes) == 0:
            return None
        segmented_frame_masks = self.__segment(
            image_source, self.sam_predictor, boxes=detected_boxes
        )
        segmented_frame_masks = segmented_frame_masks.cpu().numpy()
        segmented_frame_masks = segmented_frame_masks[:, 0, :, :]
        annotated_frame_with_mask = draw_multi_mask(
            segmented_frame_masks, annotated_frame
        )
        annotated_frame_with_mask = annotated_frame_with_mask[:, :, :3]

        detected_boxes = detected_boxes.cpu().numpy()
        segmented_frame_masks = segmented_frame_masks.astype(int)

        # RGB to BGR
        annotated_frame_with_mask = annotated_frame_with_mask[:, :, ::-1]

        return {
            "image": annotated_frame_with_mask,
            "mask": segmented_frame_masks,
            "bbox": detected_boxes,
        }

    # groundingdinoのモデルを返す
    def __load_model_hf(self, repo_id, filename, ckpt_config_filename, device="cpu"):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

        args = SLConfig.fromfile(cache_config_file)
        args.device = device
        model = build_model(args)
        model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model


    # groundingdino用の画像の前処理
    def __load_image(self, image):
        # image_pathがパス or ndarrayの場合は、PIL.Imageへ変換
        if isinstance(image, str):
            from PIL import Image

            image_source = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            from PIL import Image

            image_source = Image.fromarray(image).convert("RGB")
        else:
            image_source = image.copy()

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed


    # detect object using grounding DINO
    def __detect(
        self, image, image_source, text_prompt, model, box_threshold=0.3, text_threshold=0.25
    ):
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        annotated_frame = annotate(
            image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
        )
        annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
        return annotated_frame, boxes


    # segment object using segment anything
    def __segment(self, image, sam_model, boxes):
        sam_model.set_image(image)
        H, W, _ = image.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam_model.transform.apply_boxes_torch(
            boxes_xyxy.to(self.device), image.shape[:2]
        )
        masks, _, _ = sam_model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return masks.cpu()