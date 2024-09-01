# -*- encoding: utf-8 -*-
# @Time    :   2024/08/20 17:42:21
# @File    :   model.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   model
from torch import nn
from .utils import getClass

class Model():
    def __init__(self, config, id2label, label2id) -> None:
        params = {
            "id2label": id2label,
            "label2id": label2id,
            "ignore_mismatched_sizes": True
        }
        self.model = getClass(config["type"]).from_pretrained(**config["args"], **params)
        

        