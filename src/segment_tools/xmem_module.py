import torch
import numpy as np
import cv2
import sys
import os

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "XMem"))
from model.network import XMem as XMemNetwork
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import (
    image_to_torch,
    index_numpy_to_one_hot_torch,
    torch_prob_to_numpy_mask,
    overlay_davis,
)

from .download_weights import *

class XMem:
    """
    XMemクラスは、ビデオフレームに対するセグメンテーションを行うためのクラスです。
    """

    def __init__(self, config=None, use_BL30K=False):
        """
        XMemクラスのコンストラクタ。

        Args:
            config (dict, optional): ネットワークの設定を含む辞書。デフォルトはNone。
        """
        torch.set_grad_enabled(False)
        self.config = config or {
            "top_k": 30,
            "mem_every": 5,
            "deep_update_every": -1,
            "enable_long_term": True,
            "enable_long_term_count_usage": True,
            "num_prototypes": 128,
            "min_mid_term_frames": 5,
            "max_mid_term_frames": 10,
            "max_long_term_elements": 10000,
        }
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weight_path = "weights/XMem-s012.pth" if use_BL30K else "weights/XMem.pth"

        if not os.path.exists(weight_path):
            download_weights_xmem(weight_path, use_BL30K)

        self.network = XMemNetwork(self.config, weight_path).eval().to(self.device)
        self.processor = InferenceCore(self.network, config=self.config)

    def __process_frame(self, frame, mask_torch=None):
        frame_torch, _ = image_to_torch(frame, device=self.device)
        prediction = self.processor.step(frame_torch, mask_torch)
        return torch_prob_to_numpy_mask(prediction)
    
    def __call__(self, video_path, mask, output_video_path="output_video.mp4", no_video=False, visualize_every=1, start_frame=0, end_frame=None):
        return self.run(video_path, mask, output_video_path, no_video, visualize_every, start_frame, end_frame)

    def run(self, video_path, mask, output_video_path="output_video.mp4", no_video=False, visualize_every=1, start_frame=0, end_frame=None):
        """
        ビデオ全体を処理し、セグメンテーションマスクを生成します。
        Args:
            video_path (str): 入力ビデオのパス。
            mask (numpy.ndarray): 初期マスク。
            output_video_path (str, optional): 出力ビデオのパス。デフォルトは"output_video.mp4"。
            no_video (bool, optional): ビデオを保存しない場合はTrue。デフォルトはFalse。
            visualize_every (int, optional): 可視化の頻度。デフォルトは1。
            start_frame (int, optional): 処理を開始するフレーム番号。デフォルトは0。
            end_frame (int, optional): 処理を終了するフレーム番号。デフォルトはNone。
        Returns:
            dict: セグメンテーションマスクを含む辞書。
        """
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        # np.maximum.reduceを使って複数のマスクを結合
        combined_mask = np.maximum.reduce(mask)
        # ユニークな値を連番にマッピングする辞書を作成
        value_to_index = {v: i for i, v in enumerate(np.unique(combined_mask))}
        # マスクの各値を連番に変換
        mask = np.vectorize(value_to_index.get)(combined_mask)
        num_objects = len(np.unique(mask)) - 1
        torch.cuda.empty_cache()
        self.processor.set_all_labels(range(1, num_objects + 1))
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = end_frame or total_frames
        out = None
        if not no_video:
            frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
        output_masks = []
        current_frame_index = 0

        with Progress(
                SpinnerColumn(),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("[progress.description]{task.description}", justify="right"),
        ) as progress:
            task = progress.add_task("[green]Processing video...", total=end_frame - start_frame)
            
            with torch.cuda.amp.autocast(enabled=True):
                ret, frame = cap.read()
                if ret:
                    mask_torch = index_numpy_to_one_hot_torch(mask, num_objects + 1).to(self.device)
                    prediction = self.__process_frame(frame, mask_torch[1:])
                    output_masks.append(prediction)
                    if not no_video:
                        out.write(overlay_davis(frame, prediction))
                    current_frame_index += 1
                    progress.update(task, advance=1)
                
                while cap.isOpened() and current_frame_index < end_frame - start_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    prediction = self.__process_frame(frame)
                    output_masks.append(prediction)
                    if current_frame_index % visualize_every == 0 and not no_video:
                        out.write(overlay_davis(frame, prediction))
                    current_frame_index += 1
                    progress.update(task, advance=1)
        
        cap.release()
        if out:
            out.release()
        return {"mask": output_masks}