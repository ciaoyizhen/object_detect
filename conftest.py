# -*- encoding: utf-8 -*-
# @Time    :   2024/08/16 15:21:33
# @File    :   conftest.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   pytest add custom params


import pytest
import yaml


def pytest_addoption(parser):
    # first add extra params in ini
    parser.addini("yaml_file", "Path to the YAML file")

@pytest.fixture(scope="session")
def yaml_data(pytestconfig):
    # second get extra params in ini
    yaml_file = pytestconfig.getini("yaml_file")
    with open(yaml_file, 'r', encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


