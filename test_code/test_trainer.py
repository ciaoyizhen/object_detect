# -*- encoding: utf-8 -*-
# @Time    :   2024/08/17 14:51:03
# @File    :   test_trainer.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   test model

# import torch
# from src.trainer import ImgTrainer
# from src.utils import loadLabelFile
# from datasets import set_caching_enabled
# from torch.utils.data import DataLoader
# from transformers import DefaultDataCollator

# set_caching_enabled(False)

# def test_model_dataset(yaml_data):
#     id2label, label2id = loadLabelFile(yaml_data)
#     trainer = ImgTrainer(yaml_data, id2label, label2id)
#     dataloader = DataLoader(trainer.train_dataset.data, batch_size=yaml_data["train_args"]["train_batch_size"], collate_fn=DefaultDataCollator())
#     for batch in dataloader:
#         img = batch["pixel_values"]
#         labels = batch["labels"]
#         assert isinstance(labels, torch.Tensor)
#         assert labels.dim() == 1
#         assert isinstance(img, torch.Tensor)
#         assert img.dim() == 4
#         outputs = trainer.model.model(img)
#         logits = outputs.logits
#         assert isinstance(logits, torch.Tensor)
#         assert logits.shape[-1] == len(id2label)
#         break
    