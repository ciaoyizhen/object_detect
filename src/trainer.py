# -*- encoding: utf-8 -*-
# @Time    :   2024/08/20 17:32:49
# @File    :   trainer.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   train

import os
import logging
import numpy as np
import torch
from .model import Model
from .utils import getClass, convertAnnotationFormat
from dataclasses import dataclass
from functools import partial
from datasets import disable_caching, enable_caching
from transformers import Trainer, TrainingArguments, DefaultDataCollator
from torchmetrics.detection.mean_ap import MeanAveragePrecision 

logger = logging.getLogger("transformers.trainer")  # Unified for transformers


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor

class ObjectTrainer():
    def __init__(self, config, id2label, label2id) -> None:
        """init, used to instantiate the yaml file

        Args:
            config (dict): yaml dict
            id2label (dict): map
            label2id (dict): map
        """
        self.config = config
        self.id2label = id2label
        self.label2id = label2id
        self._initProcessor()
        self._initDataset()
        self._initModel()
        self._initTrainArgument()
        self._createTrainer()
    
    def _initProcessor(self):
        """create ImageProcessor
        """
        MAX_SIZE = 480
        params = {
            "do_resize":True,
            "size": {"max_height": MAX_SIZE, "max_width": MAX_SIZE},
            "do_pad": True,
            "pad_size": {"height": MAX_SIZE, "width": MAX_SIZE},
        }
        self.processor = getClass(self.config["processor"]["type"]).from_pretrained(**self.config["processor"]["args"], **params)
        
    def _initDataset(self):
        """init dataset
        """
        # use dataset cache
        is_dataset_cached = self.config.get("is_dataset_cached", True)
        if is_dataset_cached:
            enable_caching()
            dataset_cache_dir = self.config.get("dataset_cache_dir", None)
            if dataset_cache_dir is None:
                logger.info("dataset_cache_dir not be set, using default")
            else:
                os.environ["HF_DATASETS_CACHE"] = dataset_cache_dir
        else:
            disable_caching()
            
        
        train_cfg = self.config.get("train_dataset", None)
        validation_cfg = self.config.get("validation_dataset", None)
        test_cfg = self.config.get("test_dataset", None)
        
        assert train_cfg is not None, "train_dataset needs to be configured in the config yaml file"
        self.train_dataset = getClass(train_cfg["type"])(train_cfg["args"], self.id2label, self.label2id, self.processor)
        
        if validation_cfg is None:
            self.validation_dataset = self.train_dataset
            logger.info("validation_dataset not be set, using train_dataset")
        else:
            self.validation_dataset = getClass(validation_cfg["type"])(validation_cfg["args"], self.id2label, self.label2id, self.processor)
        
        if test_cfg is None:
            self.test_dataset = self.validation_dataset
            logger.info("test_dataset is not be set, using validation_dataset")
        else:
            self.test_dataset = getClass(test_cfg["type"])(test_cfg["args"], self.id2label, self.label2id, self.processor)

    def _initModel(self):
        self.model = Model(self.config["model"], self.id2label, self.label2id)
        
    def _initTrainArgument(self):
        args = self.config["train_args"]
        if not args.get("output_dir", None):
            args["output_dir"] = os.path.join("outputs", self.config["name"])
        
        self.train_args = TrainingArguments(**args)
    
    
    def _createTrainer(self):
        
        @torch.no_grad()
        def computeMetrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
            """
            Compute mean average mAP, mAR and their variants for the object detection task.

            Args:
                evaluation_results (EvalPrediction): Predictions and targets from evaluation.
                threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
                id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

            Returns:
                Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
            """

            predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

            # For metric computation we need to provide:
            #  - targets in a form of list of dictionaries with keys "boxes", "labels"
            #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

            image_sizes = []
            post_processed_targets = []
            post_processed_predictions = []

            # Collect targets in the required format for metric computation
            for batch in targets:
                # collect image sizes, we will need them for predictions post processing
                batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
                image_sizes.append(batch_image_sizes)
                # collect targets in the required format for metric computation
                # boxes were converted to YOLO format needed for model training
                # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
                for image_target in batch:
                    boxes = image_target["boxes"]
                    box_list = []
                    for box in boxes:
                        box = convertAnnotationFormat(box, self.config["train_dataset"]["args"]["output_format"], "voc")  # convert to voc, MeanAveragePrecision use xyxy
                        box_list.append(box)
                    boxes = torch.tensor(box_list)
                    height, width = image_target["orig_size"]
                    boxes = boxes * torch.tensor([[width, height, width, height]])
                    labels = torch.tensor(image_target["class_labels"])
                    post_processed_targets.append({"boxes": boxes, "labels": labels})

            # Collect predictions in the required format for metric computation,
            # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
            for batch, target_sizes in zip(predictions, image_sizes):
                batch_logits, batch_boxes = batch[1], batch[2]
                output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
                post_processed_output = image_processor.post_process_object_detection(
                    output, threshold=threshold, target_sizes=target_sizes
                )
                post_processed_predictions.extend(post_processed_output)

            # Compute metrics
            metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
            metric.update(post_processed_predictions, post_processed_targets)
            metrics = metric.compute()

            # Replace list of per class metrics with separate metric for each class
            classes = metrics.pop("classes")
            map_per_class = metrics.pop("map_per_class")
            mar_100_per_class = metrics.pop("mar_100_per_class")
            for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
                class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
                metrics[f"map_{class_name}"] = class_map
                metrics[f"mar_100_{class_name}"] = class_mar

            metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

            return metrics


        evalComputeMetricsFn = partial(
            computeMetrics, image_processor=self.processor, id2label=self.id2label, threshold=0.0)
            
        def collateFn(batch):
            data = {}
            data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
            data["labels"] = [x["labels"] for x in batch]
            if "pixel_mask" in batch[0]:
                data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
            return data


        self.trainer = Trainer(
            self.model.model,
            self.train_args,
            data_collator=collateFn,
            train_dataset=self.train_dataset.data,
            eval_dataset=self.validation_dataset.data,
            # compute_metrics=evalComputeMetricsFn,  #TODO some question
            tokenizer=self.processor
        )
        
    def __call__(self):
        logger.info("training start...")
        self.trainer.train()
        logger.info("train finish. evaluating...")
        self.trainer.evaluate(self.test_dataset.data)