from http import server
from httpx import delete
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import os
from .download_weights import download_weights_SAM2
from .utils import check_image_type
import shutil
import subprocess
import cv2
import supervision as sv
import gradio as gr
from gradio_rangeslider import RangeSlider
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile

sam_pt_config_dict = {
    "tiny": ["sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"],
    "small": ["sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"],
    "base": ["sam2.1_hiera_medium.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"],
    "large": ["sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"],
}


class SAM2:
    def __init__(self, model_size="tiny", weight_dir="weights"):
        checkpoint = sam_pt_config_dict[model_size][0]
        checkpoint = os.path.join(weight_dir, checkpoint)
        if not os.path.exists(checkpoint):
            download_weights_SAM2(checkpoint, model_size)

        model_cfg = sam_pt_config_dict[model_size][1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.video_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device)
        self.image_predictor = SAM2ImagePredictor(
            build_sam2(model_cfg, checkpoint, device)
        )

    def run(
        self,
        image,
        points=None,
        labels=None,
        bbox=None,
        mask=None,
        multimask_output=False,
    ):
        image = check_image_type(image)
        self.image_predictor.set_image(image)
        masks, scores, logits = self.image_predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bbox,
            mask_input=mask,
            multimask_output=multimask_output,
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks.astype(bool),
            class_id=np.zeros(masks.shape[0], dtype=int),
        )
        
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(scene=image, detections=detections)
        
        return {"image": annotated_image, "masks": masks, "scores": scores, "logits": logits}

    def get_fourcc(self):
        """
        Check if 'avc1' is supported by cv2.VideoWriter. If supported, return 'avc1',
        otherwise return 'mp4v'.

        Returns:
            int: FourCC code for 'avc1' or 'mp4v'.
        """

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "temp_check_fourcc.mp4")
            
            # Try 'avc1'
            fourcc_avc1 = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(test_file, fourcc_avc1, 30, (640, 480))
            if writer.isOpened():
                writer.release()
                print("FourCC 'avc1' is supported by cv2.VideoWriter.")
                return fourcc_avc1

            # Fallback to 'mp4v'
            fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(test_file, fourcc_mp4v, 30, (640, 480))
            if writer.isOpened():
                writer.release()
                print("FourCC 'mp4v' is supported by cv2.VideoWriter.")
                return fourcc_mp4v

        # If neither works, raise an error
        raise RuntimeError("Neither 'avc1' nor 'mp4v' is supported by cv2.VideoWriter.")

    def run_frames(
        self,
        frames,
        points=None,
        labels=None,
        bbox=None,
        mask=None,
        output_dir="sam2_processed_video",
        fps=30,
        video_name="output.mp4",
        no_video=False,
    ):
        # フレームをtempフォルダに保存
        
        with tempfile.TemporaryDirectory() as temp_dir:
            self.extract_frames_to_dir(frames, temp_dir)
            
            state = self.video_predictor.init_state(temp_dir)
            width = frames[0].shape[1]
            height = frames[0].shape[0]
            
            if points is not None and labels is not None:
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )
            elif bbox is not None:
                bbox = np.array(bbox)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    box=bbox,
                )
            elif mask is not None:
                if mask.ndim == 3:
                    for idx, mask_ in enumerate(mask):
                        _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                            inference_state=state,
                            frame_idx=0,
                            obj_id=idx + 1,
                            mask=mask_,
                        )
                else:     
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=1,
                        mask=mask,
                    )
            else:
                raise ValueError("No input given")
            
            video_segments = {} # frame_idx: {obj_id: mask}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
        output_file = None
        if not no_video:
            fps = int(fps)
            fourcc = self.get_fourcc()

            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, video_name)
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            for idx, segments in tqdm(video_segments.items()):
                object_ids = list(segments.keys())
                masks = list(segments.values())
                masks = np.concatenate(masks, axis=0)
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                    mask=masks, # (n, h, w)
                    class_id=np.array(object_ids, dtype=np.int32),
                )
                row_frame = frames[idx]
                mask_annotator = sv.MaskAnnotator()
                output_seg_image = mask_annotator.annotate(scene=row_frame, detections=detections)
                out.write(output_seg_image)
            out.release()
        
        video_segments_reshaped = np.zeros((len(frames), height, width))
        for idx, segments in tqdm(video_segments.items()):
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            for i, object_id in enumerate(object_ids):
                video_segments_reshaped[idx, :, :] += object_id * masks[i]
                
        return {"video": output_file, "masks": video_segments_reshaped}

    def run_video(
        self,
        video,
        start_frame=0,
        end_frame=None,
        points=None,
        labels=None,
        bbox=None,
        mask=None,
        output_dir="sam2_processed_video",
    ):
        # videoのパスが存在しない場合、エラーを出力
        if not os.path.exists(video):
            raise ValueError(f"Video file not found: {video}")

        # end_frameがNoneの場合、最終フレームを取得
        if end_frame is None:
            cap = cv2.VideoCapture(video)
            end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        with tempfile.TemporaryDirectory() as temp_dir:
            # videoからフレームを抽出
            self.extract_video_to_dir(video, start_frame, end_frame, temp_dir)
            
            state = self.video_predictor.init_state(temp_dir)
            
            if points is not None and labels is not None:
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )
            elif bbox is not None:
                bbox = np.array(bbox)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    box=bbox,
                )
            elif mask is not None:
                if mask.ndim == 3:
                    for idx, mask_ in enumerate(mask):
                        _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                            inference_state=state,
                            frame_idx=0,
                            obj_id=idx + 1,
                            mask=mask_,
                        )
                else:     
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=1,
                        mask=mask,
                    )
            else:
                raise ValueError("No input given")
            
            video_segments = {} # frame_idx: {obj_id: mask}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
        cap = cv2.VideoCapture(video)
        # slider_minからslider_maxまでのフレームを取得
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = self.get_fourcc()

        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, os.path.basename(video))
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        for idx, segments in tqdm(video_segments.items()):
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks, # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            ret, row_frame = cap.read()
            mask_annotator = sv.MaskAnnotator()
            output_seg_image = mask_annotator.annotate(scene=row_frame, detections=detections)
            out.write(output_seg_image)
        out.release()
        cap.release()
        
        video_segments_reshaped = np.zeros((end_frame - start_frame, height, width))
        for idx, segments in video_segments.items():
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            for i, object_id in enumerate(object_ids):
                video_segments_reshaped[idx, :, :] += object_id * masks[i]
                
        return {"video": output_file, "masks": video_segments_reshaped}

    def extract_video_to_dir(
        self, video, start_frame, end_frame, temp_dir
    ):
        start_frame, end_frame = int(start_frame), int(end_frame)

        # ffmpegコマンドを使えるか確認し、使えない場合はcv2を使用
        if shutil.which("ffmpeg") is None:
            print("ffmpeg is not installed. Using cv2")
            cap = cv2.VideoCapture(video)
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for i in range(start_frame, end_frame):
                ret, frame = cap.read()
                cv2.imwrite(f"{temp_dir}/{str(i).zfill(5)}.jpg", frame)
            cap.release()
            print("Video frames extracted")
        else:
            print("ffmpeg is installed. Using ffmpeg")
            command = f"ffmpeg -i {video} -vf 'select=between(n\,{int(start_frame)}\,{int(end_frame)})' -vsync vfr -q:v 2 {temp_dir}/%05d.jpg"
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)
            print("Video frames extracted")
            
    def extract_frames_to_dir(self, frames, temp_dir):
        for idx, frame in enumerate(frames):
            cv2.imwrite(f"{temp_dir}/{str(idx).zfill(5)}.jpg", frame)
        print("Frames extracted")

    def run_gradio(self, server_port=7860):
        self.points = np.empty((0, 2), dtype=np.float32)
        self.labels = np.empty((0,), dtype=np.int32)
        self.masks = None
        
        with gr.Blocks() as demo:
            label_type = gr.Radio(["positive", "negative"], label="SAM Label Type", value="positive")
            image = gr.Image(type="numpy", label="Input Image")
            input_image = gr.Image(type="numpy", visible=False)
                
            delete_btn = gr.Button("最後のポイントを削除")
            reset_btn = gr.Button("すべてのポイントを削除")
            process_image = gr.Image(type="numpy", label="Segmented Image")
            
            npz_file_name = gr.Textbox("", label="npzファイル名", placeholder="sam2_mask")
            save_npz_btn = gr.Button("npzとしてマスクを保存")
            npz_result = gr.Textbox("", visible=False)
            
            # 画像がアップロードされたら、input_imageに画像をセット
            image.upload(lambda x: x.copy(), [image], [input_image])
            # 画像をクリックした場合、ポイントを追加し、セグメンテーションを実行
            image.select(self.on_click, [input_image, label_type], [image, process_image])
            # ポイントの削除
            delete_btn.click(self.delete_last, [input_image], [image, process_image])
            # リセット
            reset_btn.click(self.reset, [input_image], [image, process_image])
            # npzファイルとしてマスクを保存
            save_npz_btn.click(self.save_npz, [npz_file_name], [npz_result])
            
        demo.launch(debug=True, server_port=server_port, server_name="0.0.0.0")

    def run_video_gradio(self, server_port=7860):
        self.points = np.empty((0, 2), dtype=np.float32)
        self.labels = np.empty((0,), dtype=np.int32)
        self.masks = None
        
        def process_sam2(video, range_slider):
            # ビデオセグメンテーションを実行
            start_frame, end_frame = range_slider
            result = self.run_video(video, start_frame, end_frame, self.points, self.labels)
            self.masks = result["masks"]
            
            # result["video"]の絶対パスを返す
            video_path = os.path.abspath(result["video"])
            # return result["video"]
            return gr.update(value=video_path, format="mp4")
        
        with gr.Blocks() as demo:
            video = gr.Video()
            with gr.Row():
                with gr.Column():
                    label_type = gr.Radio(["positive", "negative"], label="Label Type", value="positive")
                    range_slider = RangeSlider(minimum=0, maximum=100, value=(0, 100), visible=False)
                output_img = gr.Image(type="numpy", label="Processed Image")
                
            output_img_seg = gr.Image(type="numpy", label="Processed Image Seg", interactive=False)
            orig_img = gr.Image(type="numpy", visible=False) # 画像の元の状態を保持(ポイント付きの画像は使えないため)
            execute_btn = gr.Button("ビデオセグメンテーションを実行")
            delete_btn = gr.Button("最後のポイントを削除")
            reset_btn = gr.Button("すべてのポイントを削除")
            output_video = gr.Video()
            
            with gr.Row():
                with gr.Column():
                    npz_file_name = gr.Textbox("", label="npzファイル名", placeholder="sam2_mask")
                    save_npz_btn = gr.Button("npzとしてマスクを保存")
                npz_result = gr.Textbox("", visible=False)
            
            def update_first_frame(video_path):
                frame, frame_count = self.get_n_frame(video_path, 0)
                return [frame, frame, gr.update(maximum=frame_count, value=(0, frame_count), visible=True)]
            
            def update_nframe(video_path, range_slider):
                frame, frame_count = self.get_n_frame(video_path, range_slider[0])
                return [frame, frame]
            
            # 動画がアップロードされた場合、最初のフレームを取得
            video.change(update_first_frame, [video], [output_img, orig_img, range_slider])
            # レンジスライダーの値が変更された場合、任意のフレームを取得
            range_slider.change(update_nframe, [video, range_slider], [output_img, orig_img])
            # 画像をクリックした場合、ポイントを追加
            output_img.select(self.on_click, [orig_img, label_type], [output_img, output_img_seg])
            # ビデオセグメンテーションを実行
            execute_btn.click(process_sam2, [video, range_slider], output_video)
            # ポイントの削除
            delete_btn.click(self.delete_last, orig_img, [output_img, output_img_seg])
            # リセット
            reset_btn.click(self.reset, orig_img, [output_img, output_img_seg])
            # npzファイルとしてマスクを保存
            save_npz_btn.click(self.save_npz, [npz_file_name], [npz_result])
            
        demo.launch(debug=True, server_port=server_port, server_name="0.0.0.0")
        
    def segment_single_img(self, frame):
        image = self.run(frame, self.points, self.labels)["image"]
        return image
    
    def draw_points(self, frame):
        # ポイントを描画する
        img = frame.copy()
        for (x, y), label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 5, color, -1)
        return img
    
    def on_click(self, frame, label_type, evt: gr.SelectData):
        # 画像クリックイベントの処理
        x, y = evt.index[0], evt.index[1] # クリックした座標を取得
        if label_type == "positive":
            self.points = np.append(self.points, [[x, y]], axis=0)
            self.labels = np.append(self.labels, [1])
        elif label_type == "negative":
            self.points = np.append(self.points, [[x, y]], axis=0)
            self.labels = np.append(self.labels, [0])
        return self.draw_points(frame), self.segment_single_img(frame)
    
    def delete_last(self, frame):
        # 直前のポイントを削除
        if len(self.points) > 0:
            self.points = self.points[:-1]
            self.labels = self.labels[:-1]
        return self.draw_points(frame), self.segment_single_img(frame)
    
    def reset(self, frame):
        # ポイントをリセット
        self.points = np.empty((0, 2), dtype=np.float32)
        self.labels = np.empty((0,), dtype=np.int32)
        return self.draw_points(frame), frame
    
    def save_npz(self, npz_file_name):
        # もしnpz_file_nameが空の場合、デフォルトのファイル名を設定
        if npz_file_name == "":
            npz_file_name = "sam2_mask.npz"
        if not npz_file_name.endswith(".npz"):
            npz_file_name += ".npz"
            
        # npz_file_nameがすでに存在する場合、上書きしない
        if os.path.exists(npz_file_name):
            return gr.update(value=f"{npz_file_name} already exists. Please choose another name or delete the existing file.", visible=True)
        np.savez_compressed(npz_file_name, mask=self.masks)
        return gr.update(value=f"Saved as {npz_file_name}. \nnpz_key=mask", visible=True)
    
    def get_n_frame(self, video_path, frame_num):
        # 動画から任意のフレームを取得し、任意のフレームとフレーム数を返す
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = min(frame_num, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        if not ret:
            raise ValueError("動画のフレームを取得できませんでした。")
        return frame, frame_count
