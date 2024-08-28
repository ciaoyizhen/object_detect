# -*- encoding: utf-8 -*-
# @Time    :   2024/08/25 15:54:34
# @File    :   web_demo.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   web demo

import gradio as gr
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from enum import Enum
# load_model
model = AutoModelForObjectDetection.from_pretrained("outputs/cppe-5/checkpoint-4598")
processor = AutoImageProcessor.from_pretrained("outputs/cppe-5/checkpoint-4598")

class BoxFormat(Enum):
    VOC= "voc"  # x_min, y_min, x_max, y_max
    COCO = "coco" # x_min, y_min, width, height
    YOLO = "yolo" # center_x, center_y, width, height
    POLYGON = "polygon"  # x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
    #TODO if you want a truly polygon, please implement it yourself to get max box outlines

def convertAnnotationFormat(box, input_format, output_format):
    """convert annotation box format

    Args:
        box (list[int]): image annotation box
        input_format (str): should be `BoxFormat`(enum)
        output_format (str): should be `BoxFormat`(enum)
    
    Returns:
        list: output_format box 
    """
    input_format = BoxFormat(input_format)
    output_format = BoxFormat(output_format)
        
    # transform format to POLYGON
    match input_format:
        case BoxFormat.VOC:
            x_min, y_min, x_max, y_max = box
            polygon = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
        case BoxFormat.COCO:
            x_min, y_min, width, height = box
            x_max, y_max = x_min + width, y_min + height
            polygon = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
        case BoxFormat.YOLO:
            cx, cy, width, height = box
            x_min, x_max, y_min, y_max = cx - width, cx + width, cy - height, cy + height
            polygon = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
        case BoxFormat.POLYGON:
            polygon = box
    
    match output_format:
        case BoxFormat.VOC:
            x_min, y_min, x_max, _, _, y_max, _, _ = polygon
            return [x_min, y_min, x_max, y_max]
        case BoxFormat.COCO:
            x_min, y_min, x_max, _, _, y_max, _, _ = polygon
            width, height = x_max - x_min, y_max - y_min
            return [x_min, y_min, width, height]
        case BoxFormat.YOLO:
            x_min, y_min, x_max, _, _, y_max, _, _ = polygon
            cx, cy = (x_min + x_max)/2, (y_min + y_max)/2
            width, height = (x_max - x_min)/2, (y_max - y_min)/2
            return [cx, cy, width, height]
        case BoxFormat.POLYGON:
            return polygon
            

@torch.inference_mode()
def detect_objects(image, threshold):
    
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    print(results)
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        box = [round(i, 2) for i in box.tolist()]
        # box = convertAnnotationFormat(box, "coco", "voc")
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}", fill="green")

    return image

interface = gr.Interface(
    fn=detect_objects,
    inputs=[gr.Image(type="pil"), gr.Slider(0.0, 1.0, value=0.5, label="Threshold")],
    outputs=gr.Image(type="pil"),
    title="object_detect",
    description="upload a image, return a image with detection box and label.",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=2024)