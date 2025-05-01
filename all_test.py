from PIL import Image
import numpy as np
import cv2
import segment_tools as st
import os

image_path = "cityscapes.png"
image_pil = Image.open(image_path)  # Open image with Pillow
image_np = np.array(image_pil)      # Convert to numpy array

import logging
# ルートロガーのログレベルをCRITICALに設定し、すべてのハンドラを削除
logging.getLogger().setLevel(logging.ERROR)
[logging.getLogger().removeHandler(h) for h in logging.getLogger().handlers]

import warnings
warnings.filterwarnings('ignore')

prompt = "car"  # Define your prompt

def save_image(image, filename):
    os.makedirs("test_output", exist_ok=True)
    cv2.imwrite(f"test_output/{filename}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
# clipseg
clipseg = st.CLIPSeg()
result = clipseg(image_np, prompt)
image, ann = result["image"], result["mask"]
save_image(image, "clipseg.jpg")
print("DONE: clipseg")
del clipseg, result, image, ann

# depth_anything
depth_anything = st.Depth_Anything(encoder="vitl") # vits or vitb or vitl
result = depth_anything(image_np)
image, depth = result["image"], result["depth"]
save_image(image, "depth_anything.jpg")
print("DONE: depth_anything")
del depth_anything, result, image, depth

# depth_pro
depth_pro = st.Depth_Pro()
result = depth_pro(image_np)
image, depth = result["image"], result["depth"]
save_image(image, "depth_pro.jpg")
print("DONE: depth_pro")
del depth_pro, result, image, depth

# dino
dino = st.DINO()
result = dino(image_np, prompt, xyxy_bbox=True, abs_bbox=True)
image, bbox = result["image"], result["bbox"]
save_image(image, "dino.jpg")
print("DONE: dino")
del dino, result, image, bbox

# dinoseg
dinoseg = st.DINOSeg()
result = dinoseg(image_np, prompt)
image, ann, bbox = result["image"], result["mask"], result["bbox"]
save_image(image, "dinoseg.jpg")
print("DONE: dinoseg")
del dinoseg, result, image, ann, bbox

# dinov2
dinov2 = st.DINOv2_depth()
result = dinov2(image_np)
image, depth = result["image"], result["depth"]
save_image(image, "dinov2.jpg")
print("DONE: dinov2")
del dinov2, result, image, depth

# fastsam ultralyticsに変更する
# fastsam = st.FastSAM()
# result = fastsam.run(image_np)
# image, ann = result["image"], result["mask"]
# save_image(image, "fastsam.jpg")
# print("DONE: fastsam")
# del fastsam

# florence2
florence2 = st.Florence2()
result = florence2(image_np)
image, bbox, label = result["image"], result["bbox"], result["label"]
save_image(image, "florence2.jpg")
print("DONE: florence2")
del florence2, result, image, bbox, label

# grit
grit = st.GRiT()
result = grit(image_np)
image, bbox, info = result["image"], result["bbox"], result["info"]
save_image(image, "grit.jpg")
print("DONE: grit")
del grit, result, image, bbox, info

# OneFormer
# ADE20K(dinat)
oneformer_ade20k = st.OneFormer(dataset="ade20k")
result = oneformer_ade20k(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_ade20k.jpg")
# ADE20K(dinat) with prompt
result = oneformer_ade20k(image_np, prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_ade20k_prompt.jpg")
del oneformer_ade20k, result, image, ann
# ADE20K(swin)
oneformer_ade20k_swin = st.OneFormer(dataset="ade20k", backbone="swin")
result = oneformer_ade20k_swin(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_ade20k_swin.jpg")
# ADE20K(swin) with prompt
result = oneformer_ade20k_swin(image_np, prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_ade20k_swin_prompt.jpg")
del oneformer_ade20k_swin, result, image, ann
# ADE20K(convnext)
oneformer_ade20k_conv = st.OneFormer(dataset="ade20k", backbone="convnext")
result = oneformer_ade20k_conv(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_ade20k_convnext.jpg")
# ADE20K(convnext) with prompt
result = oneformer_ade20k_conv(image_np, prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_ade20k_convnext_prompt.jpg")
print("DONE: OneFormer_ADE20K")
del oneformer_ade20k_conv, result, image, ann

# Cityscapes(dinat)
oneformer_city = st.OneFormer(dataset="cityscapes")
result = oneformer_city(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_city.jpg")
# Cityscapes(dinat) with prompt
result = oneformer_city(image_np, prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_city_prompt.jpg")
del oneformer_city, result, image, ann
# Cityscapes(swin)
oneformer_city_swin = st.OneFormer(dataset="cityscapes", backbone="swin")
result = oneformer_city_swin(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_city_swin.jpg")
# Cityscapes(swin) with prompt
result = oneformer_city_swin(image_np, prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_city_swin_prompt.jpg")
del oneformer_city_swin, result, image, ann
# Cityscapes(convnext)
oneformer_city_conv = st.OneFormer(dataset="cityscapes", backbone="convnext")
result = oneformer_city_conv(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_city_convnext.jpg")
# Cityscapes(convnext) with prompt
result = oneformer_city_conv(image_np, prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_city_convnext_prompt.jpg")
print("DONE: OneFormer_Cityscapes")
del oneformer_city_conv, result, image, ann

# COCO(dinat)
oneformer_coco = st.OneFormer(dataset="coco")
result = oneformer_coco(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_coco.jpg")
# COCO(dinat) with prompt
result = oneformer_coco(image_np, prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_coco_prompt.jpg")
del oneformer_coco, result, image, ann
# COCO(swin)
oneformer_coco_swin = st.OneFormer(dataset="coco", backbone="swin")
result = oneformer_coco_swin(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_coco_swin.jpg")
# COCO(swin) with prompt
result = oneformer_coco_swin(image_np, prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_coco_swin_prompt.jpg")
print("DONE: OneFormer_COCO")
del oneformer_coco_swin, result, image, ann

vista_prompt = "Car"
# Vistas(dinat)
oneformer_vistas = st.OneFormer(dataset="vistas")
result = oneformer_vistas(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_vistas.jpg")
# Vistas(dinat) with prompt
result = oneformer_vistas(image_np, vista_prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_vistas_prompt.jpg")
del oneformer_vistas, result, image, ann
# Vistas(swin)
oneformer_vistas_swin = st.OneFormer(dataset="vistas", backbone="swin")
result = oneformer_vistas_swin(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_vistas_swin.jpg")
# Vistas(swin) with prompt
result = oneformer_vistas_swin(image_np, vista_prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_vistas_swin_prompt.jpg")
del oneformer_vistas_swin, result, image, ann
# Vistas(convnext)
oneformer_vistas_conv = st.OneFormer(dataset="vistas", backbone="convnext")
result = oneformer_vistas_conv(image_np)
image, ann, info = result["image"], result["mask"], result["info"]
save_image(image, "oneformer_vistas_convnext.jpg")
# Vistas(convnext) with prompt
result = oneformer_vistas_conv(image_np, vista_prompt)
image, ann = result["image"], result["mask"]
save_image(image, "oneformer_vistas_convnext_prompt.jpg")
print("DONE: OneFormer_Vistas")
del oneformer_vistas_conv, result, image, ann

# XMem
cap = cv2.VideoCapture("hazard_test.mp4")
ret, frame = cap.read()
ann = st.OneFormer(dataset="cityscapes")(frame, "person")["mask"] # Use OneFormer to get the mask
# multi object tracking
xmem = st.XMem()
result = xmem("hazard_test.mp4", ann[0], "test_output/hazard_test_multi.mp4")
print("DONE: XMem_single")
# one object tracking
result = xmem("hazard_test.mp4", ann[0][0], "test_output/hazard_test_single.mp4")
print("DONE: XMem_multi")
cap.release()
del xmem, ann, result

# SAM2(1point)
sam2 = st.SAM2()
result = sam2.run(image_np, points=[[331, 516]], labels=[1])
image, ann, scores, logits = result["image"], result["masks"], result["scores"], result["logits"]
save_image(image, "sam2_1point.jpg")
del result, image, ann, scores, logits
# SAM2(2point)
result = sam2.run(image_np, points=[[331, 516], [313, 523]], labels=[1, 1])
image = result["image"]
save_image(image, "sam2_2point.jpg")
del result, image
# SAM2(2point_negative)
result = sam2.run(image_np, points=[[331, 516], [313, 523]], labels=[1, 0])
image = result["image"]
save_image(image, "sam2_2point_negative.jpg")
del result, image, sam2
print("DONE: SAM2_point")

# SAM2(bbox_multibox) 
sam2 = st.SAM2()
dino = st.DINO()
bbox = dino(image_np, "car", xyxy_bbox=True, abs_bbox=True)["bbox"]
result = sam2.run(image_np, bbox=bbox)
image = result["image"]
save_image(image, "sam2_bbox_multi.jpg")
# SAM2(bbox_singlebox)
bbox = dino(image_np, "car", xyxy_bbox=True, abs_bbox=True)["bbox"][0]
result = sam2.run(image_np, bbox=bbox)
image = result["image"]
save_image(image, "sam2_bbox_single.jpg")
del sam2, dino, result, image
print("DONE: SAM2_bbox")

# SAM2(frames_point)
sam2 = st.SAM2()
cap = cv2.VideoCapture("hazard_test.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
result = sam2.run_frames(frames, points=[[999, 579]], labels=[1])
print(result["video"])
del frames, result, sam2
cap.release()
print("DONE: SAM2_frames")

# SAM2(video_point)
sam2 = st.SAM2()
result = sam2.run_video("hazard_test.mp4", start_frame=0, end_frame=None, points=[[999, 579]], labels=[1])
print(result["video"])
del result, sam2
print("DONE: SAM2_video")
