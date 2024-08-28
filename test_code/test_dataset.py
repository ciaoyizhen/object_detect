# -*- encoding: utf-8 -*-
# @Time    :   2024/08/16 15:41:07
# @File    :   test_dataset.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   test dataset
import torch
from src.dataset import Dataset
from src.utils import loadLabelFile
from datasets import set_caching_enabled
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

set_caching_enabled(False)

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data

def test_single_dataset(yaml_data):
    processor = AutoImageProcessor.from_pretrained("weights/facebook-detr-resnet-50")
    id2label, label2id = loadLabelFile(yaml_data)
    dataset = Dataset(yaml_data["train_dataset"]["args"], id2label, label2id, processor)
    dataloader = DataLoader(dataset.data, batch_size=yaml_data["train_args"]["train_batch_size"], collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        img = batch["pixel_values"]
        labels = batch["labels"]
        assert isinstance(labels, torch.Tensor)
        assert labels.dim() == 1
        assert isinstance(img, torch.Tensor)
        assert img.dim() == 4
        break