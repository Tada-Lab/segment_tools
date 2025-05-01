from .FastSAM.fastsam import FastSAM as fs, FastSAMPrompt
import torch
import cv2
import os


class FastSAM:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_path="weights/FastSAM.pt"
        if not os.path.exists(model_path):
            from .download_weights import download_weights_FastSAM
            download_weights_FastSAM(model_path)
        self.model = fs(model_path)

    def run(
        self,
        image_path,
        text=None,
        points=None,
        point_labels=None,
        bboxes=None,
        bbox_labels=None,
    ):
        """FastSAMを用いた画像のセグメンテーション
        Args:
            image_path: PILでもnumpyでもパスでも可
            text: テキストプロンプト. Defaults to None.
            points: ポイントプロンプト. points default [[0,0]] or [[x1,y1],[x2,y2]] など. Defaults to None.
            point_labels: 背景か前景か. point_label default [0] or [1,0] or [1] 0:背景, 1:前景. Defaults to None.
            bboxes: ボックスプロンプト.bbox default shape [0,0,0,0] -> [x1,y1,x2,y2].  Defaults to None.
            bbox_labels (_type_, optional): _description_. Defaults to None.

        Returns:
            image: 結果の画像
            mask: セグメンテーション結果のマスク
        """
        IMAGE_PATH = image_path
        everything_results = self.model(
            IMAGE_PATH,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9,
        )
        prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=self.device)

        # everything prompt
        ann = prompt_process.everything_prompt()
        ann = ann.cpu().numpy()
        ann = ann.astype(int)

        if bboxes is not None:
            # # bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
            ann = prompt_process.box_prompt(bboxes=bboxes, boxlabel=bbox_labels)

        if text is not None:
            # text prompt
            ann = prompt_process.text_prompt(text=text)
            ann = ann.astype(int)

        if points is not None:
            # point prompt
            # points default [[0,0]] [[x1,y1],[x2,y2]]
            # point_label default [0] [1,0] 0:background, 1:foreground
            ann = prompt_process.point_prompt(points=points, pointlabel=point_labels)

        output_image = prompt_process.plot_to_result(annotations=ann)

        # RGB to BGR
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        return {"image": output_image, "mask": ann}
