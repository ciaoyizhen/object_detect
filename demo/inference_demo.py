# -*- encoding: utf-8 -*-
# @Time    :   2024/08/17 21:26:07
# @File    :   inference_demo.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   inference_demo
"""
A small operation is needed here
Because the model is saved without saving the preprocess
So you need to manually put the preprocess_config.json to the location where the model was saved.
"""
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoModelForObjectDetection, AutoImageProcessor

img_path = "data/train/train_imgs/0.jpg"
image = Image.open(img_path)
weight_path = "outputs/cppe-5/checkpoint-1600"
processor = AutoImageProcessor.from_pretrained(weight_path)  #! need preprocessor_config.json
model = AutoModelForObjectDetection.from_pretrained(weight_path)
iou_threshold = 0.7  # for nms

model.eval()
with torch.inference_mode():
    inputs = processor(images=[image], return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([[image.size[1], image.size[0]]])
    results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]

#! nms   may this is not need
boxes = results["boxes"]
scores = results["scores"]
labels = results["labels"]
index = nms(boxes, scores, iou_threshold=iou_threshold).tolist()

for i in index:
    score = scores[i]
    box = boxes[i]
    label = labels[i]
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

