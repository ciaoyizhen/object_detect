# -*- encoding: utf-8 -*-
# @Time    :   2024/08/18 12:14:53
# @File    :   download_dataset.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   download datasets
import os
from datasets import load_dataset
from src.utils import convertAnnotationFormat
from tqdm import tqdm



cppe5 = load_dataset("cppe-5")  #! just down example dataset, you can use any datasets
print(cppe5)
label_name = cppe5["train"].features["objects"].feature["category"].names

with open("configs/label_config.txt", "w", encoding="utf-8") as f:
    for label in label_name:
        f.write(label + "\n")            


def saveDataset(split:str):
    config_data_path = f"configs/{split}_data.txt"
    f = open(config_data_path, "w", encoding="utf-8")
    image_save_dir = f"data/{split}/{split}_imgs"
    annotations_save_dir = f"data/{split}/{split}_annotations"

    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(annotations_save_dir, exist_ok=True)

    for idx, sample in tqdm(enumerate(cppe5[split]), total=len(cppe5[split])):
        try:
            image = sample["image"]
            # if image.mode != "RGB":
            #     image = image.convert("RGB")
            
            objects = sample["objects"]
            bbox = objects["bbox"]
            category = objects["category"]
            img_path = os.path.join(image_save_dir, f"{idx}.jpg")
            image.save(img_path)
            annotation_path = os.path.join(annotations_save_dir, f"{idx}.txt")
            annotations = open(annotation_path, "w", encoding="utf-8")
            for box, label_id in zip(bbox, category):
                box = convertAnnotationFormat(box, "coco", "polygon")
                box = list(map(str, box))
                label = label_name[label_id]
                box = ",".join(box)
                annotations.write(f"{box}\t{label}\n")
            annotations.close()
            f.write(img_path + "\t" + annotation_path + "\n")
        except:
            continue
    f.close()
    
saveDataset("train")
saveDataset("test")

