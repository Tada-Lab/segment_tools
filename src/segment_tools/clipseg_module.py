from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch

import numpy as np
from PIL import Image
import cv2
from .utils import separate_masks, draw_multi_mask, check_image_type


class CLIPSeg:
    def __init__(self, model_name="CIDAS/clipseg-rd64-refined"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device)
        
    def __call__(self, image, text, threshold=100, color="random"):
        return self.run(image, text, threshold, color)

    def run(self, image, text, threshold=100, color="random"):
        """clipsegを用いた画像のセグメンテーション

        Args:
            image: PILでもnumpyでもパスでも可
            text: テキストプロンプト

        Returns:
            image: セグメンテーション結果を描画した画像
            masks: セグメンテーション結果のマスク
        """
        image = check_image_type(image, type="pil")

        image_shape = (image.width, image.height)
        # 画像のリサイズとRGB変換
        image_ = image.resize((352, 352))
        image_ = image.convert("RGB")
        # プロンプトの準備
        prompts = [text]
        # 推論の準備
        inputs = self.processor(
            text=prompts,
            images=[image_] * len(prompts),
            padding="max_length",
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        # 推論
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 確率を取得(352x352)
        preds = outputs.logits
        # 確率をsigmoidにかける(正規化?)
        output = torch.sigmoid(preds)
        # torchからnumpyへ
        output_np = output.detach().cpu().numpy().squeeze()
        # 0-255へ
        output_np = Image.fromarray((output_np * 255).astype(np.uint8))
        # 元の画像サイズへ
        output_np = output_np.resize(image_shape)
        # PILからnumpyへ
        output_np = np.array(output_np)
        # 二値化
        output_image = cv2.threshold(output_np, threshold, 255, cv2.THRESH_BINARY)[1]

        # マスクの分離
        masks = separate_masks(output_image)
        image = np.array(image)
        # マスクの描画
        if masks.shape[0] == 0:
            print("No mask found.")
            return None
        drawed_mask = draw_multi_mask(masks, image, text, color=color)

        drawed_mask = drawed_mask[:, :, :3]

        # RGB to BGR
        drawed_mask = drawed_mask[:, :, ::-1]

        return {"image": drawed_mask, "mask": masks}
