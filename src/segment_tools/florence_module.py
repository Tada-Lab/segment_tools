import torch
from transformers import AutoProcessor, AutoModelForCausalLM 
from PIL import Image
import cv2
import supervision as sv

from .utils import check_image_type

class Florence2:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        ).to("cuda:0")
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", 
            trust_remote_code=True
        )
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup = sv.ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(color_lookup = sv.ColorLookup.INDEX)
        
    def __call__(self, image, task_prompt="<DENSE_REGION_CAPTION>", no_image=False, text_input=None):
        return self.run(image, task_prompt, no_image, text_input)
        
    def run(self, image, task_prompt="<DENSE_REGION_CAPTION>", no_image=False, text_input=None):
        image = check_image_type(image, "pil")
        
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda:0", torch.float16)
        generated_ids = self.model.generate(
          input_ids=inputs["input_ids"],
          pixel_values=inputs["pixel_values"],
          max_new_tokens=1024,
          num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

        if no_image:
            return {"image": None, "bbox": result[task_prompt]['bboxes'], "label": result[task_prompt]['labels']}
        else:
            detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, result, resolution_wh=image.size)
            out_image = self.bounding_box_annotator.annotate(image.copy(), detections)
            out_image = self.label_annotator.annotate(out_image, detections)
            out_image = check_image_type(out_image)
            return {"image": out_image, "bbox": result[task_prompt]['bboxes'], "label": result[task_prompt]['labels']}
        return result