# -*- encoding: utf-8 -*-
# @Time    :   2024/08/18 12:51:08
# @File    :   utils.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   utils
import importlib
from enum import Enum
from typing import Tuple, Dict, Type

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
            
def calculateArea(box:list, input_format:BoxFormat)->int:
    """calculate area by given box

    Args:
        box (list): box points
        input_format (_type_): box format, BoxFormat

    Returns:
        int: box area
    """
    input_format = BoxFormat(input_format)
    
    match input_format:
        case BoxFormat.VOC:
            x_min, y_min, x_max, y_max = box
            area = (x_max - x_min) * (y_max - y_min)
        case BoxFormat.COCO:
            x_min, y_min, width, height = box
            area = width * height
        case BoxFormat.YOLO:
            center_x, center_y, width, height = box
            area = width * height
        case BoxFormat.POLYGON:
            x_min, y_min, x_max, _, _, y_max, _, _ = box
            area = (x_max - x_min) * (y_max - y_min)
    return area

def normalizeBox(box, image_size, input_format):
    """normalize Box from true size to [0, 1]
    
    Args:
        box (list): input box
        image_size (list): width, height  original image size
        input_format (str): should be `BoxFormat`(enum)
    
    Returns:
        list: normalize box 
    """
    
    input_format = BoxFormat(input_format)
    coco_box = convertAnnotationFormat(box, input_format, "coco")
    x_min, y_min, width, height = coco_box
    img_w, img_h = image_size
    resize_box = [x_min/img_w, y_min/img_h, width/img_w, height/img_h]
    box = convertAnnotationFormat(resize_box, "coco", input_format)
    return box


def getClass(module:str) -> Type:
    """return class based on module

    Args:
        module (str): format `module,class`

    Returns:
        Type: The class object corresponding to the specified module and class name.
    
    Example:
        >>> MyClass = getClass("my_module,MyClass")
        >>> instance = MyClass()
        >>> print(isinstance(instance, MyClass))
        True
    """
    module, class_ = module.split(",")
    module = importlib.import_module(module)
    class_ = getattr(module, class_)
    return class_


def loadLabelFile(config:dict) -> Tuple[Dict[int, str], Dict[str, int]]:
    """generate id2label and label2id based on label_config in yaml
    Args:
        config (dict): config yaml dict
        
    Returns:
        Tuple[Dict[int, str], Dict[str, int]]:
            - id2label (Dict[int, str]): int -> label
            - label2id (Dict[str, int]): label -> int

    Example:
        >>> config = {"label_config": "path/to/label_config.txt"}
        >>> id2label, label2id = loadLabelFile(config)
        >>> print(id2label)
        {0: 'label1', 1: 'label2'}
        >>> print(label2id)
        {'label1': 0, 'label2': 1}
    """
    id2label = {}
    label2id = {}
    id_ = 0
    with open(config["label_config"], "r", encoding="utf-8") as f:
        for line in f.readlines():
            if label := line.strip():
                id2label[id_] = label
                label2id[label] = id_
                id_ += 1
                
    return id2label, label2id