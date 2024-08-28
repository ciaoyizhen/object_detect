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
        
        # self.model = getClass(config["type"]).from_pretrained(**config["args"])
        # try:
        #     #! DetrForObjectDetection example, you should process your model yourself
        #     self.model.config.num_labels = len(id2label) + 1
        #     self.model.class_labels_classifier = nn.Linear(self.model.config.d_model, self.model.config.num_labels)
        # except:
        #     print("Your model is different from the example and needs to be manually adjusted to the source code")
        # self.model.config.id2label = id2label
        # self.model.config.label2id = label2id
        