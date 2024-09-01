# -*- encoding: utf-8 -*-
# @Time    :   2024/08/18 16:20:17
# @File    :   dataset.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   data

import os
import numpy as np
import torch
import albumentations as A
from PIL import Image
from datasets import load_dataset, concatenate_datasets
from .utils import convertAnnotationFormat, calculateArea, getClass, BoxFormat


class Dataset():
    def __init__(self, cfg, id2label, label2id, processor) -> None:
        """init

        Args:
            cfg (dict): args of **_dataset in yaml
            id2label (dict):
            label2id (dict):
            processor (Processor): transformers.AutoImageProcessor
        """
        
        self.cfg = cfg
        self.id2label = id2label
        self.label2id = label2id
        self.processor = processor
        self.format_map = {
            ".txt": "text",
            ".csv": "csv",
            ".json": "json",
            ".tsv": "csv"
        }
        self._loadData()  # str format not Tensor
        self._createTransform()
        self._process()

    def _chooseFileFormat(self, file_path:str) -> str:
        """get function `load_dataset` format based on file_path

        Args:
            file_path (str): input_file_path
        
        Returns:
            str: function `load_dataset` format

        Example:
            >>> format_ = self._chooseFileFormat("data/file.txt")
            >>> print(format_)
            "text"
        """
        _, ext = os.path.splitext(file_path)
        format_ = self.format_map.get(ext, None)
        assert format_ is not None, f"file_path only support {set(self.format_map.keys())}"
        if format_ not in {"text"}:
            raise NotImplementedError(f"currently not implement {ext}")
        return format_

    def _loadData(self):
        print("loading Data...")
        data_path_list = self.cfg["data_paths"]
        assert len(data_path_list) != 0, "data_paths length not be zero!"
        
        if len(data_path_list) == 1:
            data_path = data_path_list[0]
            format_ = self._chooseFileFormat(data_path)
            self.data = load_dataset(format_, data_files=data_path, split="train")
        else:        
            datasets = []
            for data_path in data_path_list:
                format_ = self._chooseFileFormat(data_path)
                datasets.append(load_dataset(format_, data_files=data_path, split="train"))
            self.data = concatenate_datasets(datasets)

    def _createTransform(self):
        if "albumentations" in self.cfg:
            albumentations_ = self.cfg["albumentations"]
        else:
            self.transform = None
            return
        A_list = []
        for item in albumentations_:
            kwargs = item.get("args", None)
            module_ = getClass(item["type"])
            if kwargs is not None:
                class_ = module_(**kwargs)
            else:
                class_ = module_()
            A_list.append(class_)
        output_format = BoxFormat(self.cfg["output_format"])
        format_ = output_format.ConvertAlbumentationsFormat()
        self.transform = A.Compose(A_list, bbox_params=A.BboxParams(format=format_, label_fields=['class_labels'], min_area=self.cfg["min_area"]))

    def _process(self):

        def transformAnn(examples):
            text = examples["text"]
            image_id = examples["image_id"]

            pixel_values_list = []
            # pixel_mask_list = []
            labels_list = []
            for item, id_ in zip(text, image_id):
                img_path, label_path = item.split("\t")
                try:
                    img = Image.open(img_path)
                    img = img.convert("RGB")
                    if self.transform is not None:
                        img = np.asarray(img)
                    annotation = []
                    bbox_list = []
                    category_list = []
                    with open(label_path, "r", encoding="utf-8") as f:
                        for line in f.readlines():
                            if line := line.strip():
                                points, label = line.split("\t")
                                points = points.split(",")
                                bbox = list(map(lambda x: int(float(x)), points))
                                bbox = convertAnnotationFormat(bbox, input_format=self.cfg["input_format"], output_format=self.cfg["output_format"])
                                area = calculateArea(bbox, self.cfg["output_format"])
                                # label = self.label2id[label]
                                ann = {
                                    "image_id": id_,
                                    # "category_id": label,
                                    "iscrowd": 0,
                                    "area": area,
                                    # "bbox": bbox,
                                }
                                bbox_list.append(bbox)
                                category_list.append(label)
                                annotation.append(ann)
                    
                    if self.transform is None:
                        # if not do data enhancement
                        new_annotation = []
                        for i, ann in enumerate(annotation):
                            ann["category_id"] = self.label2id[category_list[i]]
                            ann["bbox"] = bbox_list[i]
                            new_annotation.append(ann)
                    # data enhancement
                    else:
                        transformed_data = self.transform(image=img, bboxes=bbox_list, class_labels=category_list)
                        img = transformed_data["image"]
                        transform_box = transformed_data["bboxes"]
                        transformed_labels = transformed_data["class_labels"]
                        new_annotation = []
                        for i, ann in enumerate(annotation):
                            ann["category_id"] = self.label2id[transformed_labels[i]]
                            ann["bbox"] = transform_box[i]
                            new_annotation.append(ann)
                        
                    
                    images = [img]
                    targets = [{"image_id": id_, "annotations": new_annotation}]
                    single_input = self.processor(images=images, annotations=targets, return_tensors="pt")
                    # remove batch
                    pixel_values_list.append(single_input["pixel_values"].squeeze(0))
                    # pixel_mask_list.append(single_input["pixel_mask"].squeeze(0))  # because our inputs is fixed size, this is not need.
                    labels_list.append(single_input["labels"][0])
                except:
                    continue
            # check value
            if not pixel_values_list:
                return None
            
            return {"pixel_values": torch.stack(pixel_values_list), "labels": labels_list}
            # return {"pixel_values": torch.stack(pixel_values_list), "pixel_mask": torch.stack(pixel_mask_list), "labels": labels_list}
        
        def addImageId(example, idx):
            example["image_id"] = idx
            return example
        
        self.data = self.data.map(addImageId, with_indices=True)  # each image add image_id
        self.data = self.data.with_transform(transformAnn)
        
