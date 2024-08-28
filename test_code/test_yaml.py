# -*- encoding: utf-8 -*-
# @Time    :   2024/08/16 15:06:43
# @File    :   test_yaml.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   test yaml file



def test_YamlFile(yaml_data):
    assert yaml_data is not None
    assert isinstance(yaml_data, dict)