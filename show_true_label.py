# -*- encoding: utf-8 -*-
# @Time    :   2024/08/18 14:46:44
# @File    :   test.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   show true object position

import os
from src.utils import convertAnnotationFormat
from PIL import Image, ImageDraw, ImageFont


img_path = "data/train/train_imgs/0.jpg"
txt_path = "data/train/train_annotations/0.txt"
image = Image.open(img_path)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("/System/Library/Fonts/Apple Symbols.ttf", size=40)
with open(txt_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        if line:= line.strip():
            points, label = line.split("\t")
            points = points.split(",")
            points = list(map(lambda x: int(float(x)), points))
            box = convertAnnotationFormat(points, "polygon", "voc")
            draw.rectangle(box, outline="red", width=2)
            draw.text(box, label, font=font, fill=(0, 255, 0))
image.show()
