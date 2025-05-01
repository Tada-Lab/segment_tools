from PIL import Image

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, predict
import numpy as np

# diffusers
import torch
from huggingface_hub import hf_hub_download
from .utils import check_image_type
import cv2

class DINO:
    def __init__(
        self,
        ckpt_repo_id="ShilongLiu/GroundingDINO",
        ckpt_filenmae="groundingdino_swinb_cogcoor.pth",
        ckpt_config_filename="GroundingDINO_SwinB.cfg.py",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.groundingdino_model = self.__load_model_hf(
            ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, self.device
        )
        
    def __call__(self, image, text, xyxy_bbox=False, abs_bbox=False):
        return self.run(image, text, xyxy_bbox, abs_bbox)

    def run(self, image, text, xyxy_bbox, abs_bbox):
        """DINOを用いた画像のゼロショット物体検出

        Args:
            image: PILでもnumpyでもパスでも可
            text: テキストプロンプト

        Returns:
            image: 物体検出結果
            bbox: 物体検出結果のバウンディングボックス
        """
        image_source, image = self.__load_image(image)
        annotated_frame, detected_boxes = self.__detect(
            image, image_source, text_prompt=text, model=self.groundingdino_model
        )
        # detected_boxesをtensorからndarrayに変換
        detected_boxes = detected_boxes.cpu().numpy()

        # RGB to BGR
        annotated_frame = annotated_frame[:, :, ::-1]

        if xyxy_bbox:
            # detected_boxesはbbox中央のxyと幅高さの情報を持っているので、xyxy形式に変換
            detected_boxes[:, 0] -= detected_boxes[:, 2] / 2
            detected_boxes[:, 1] -= detected_boxes[:, 3] / 2
            detected_boxes[:, 2] += detected_boxes[:, 0]
            detected_boxes[:, 3] += detected_boxes[:, 1]
            
        if abs_bbox:
            # bounding boxの座標を絶対座標に変換
            height, width, _ = annotated_frame.shape
            detected_boxes[:, 0] *= width
            detected_boxes[:, 1] *= height
            detected_boxes[:, 2] *= width
            detected_boxes[:, 3] *= height

        return {"image": annotated_frame, "bbox": detected_boxes}
    
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
        image_source = check_image_type(image, "pil")
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
