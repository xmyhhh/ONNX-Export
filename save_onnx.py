from collections import OrderedDict

import os
import logging
import torch
import argparse
from logging import handlers
import yaml
import torch.onnx
import onnx
from Tool.File import create_all_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SR results")
    parser.add_argument('YAML', type=str, help='configuration file')
    args = parser.parse_args()

    conf = dict()
    with open(args.YAML, 'r', encoding='UTF-8') as f:
        conf = yaml.load(f.read(), Loader=yaml.FullLoader)

    ModelDefinePyPath = conf['ModelDefinePyPath']
    ModelDefineClassName = conf['ModelDefineClassName']
    ModelPthPath = conf['ModelPthPath']
    SavePath = conf['SavePath']

    for idx, item in enumerate(ModelDefinePyPath):
        create_all_dirs(SavePath[idx])

        import_string = "from Wrapper." + str(ModelDefinePyPath[idx]) + " import " + str(
            ModelDefineClassName[idx]) + " as ModelBuilder"

        exec(import_string)

        model = ModelBuilder() # 编译器报错正常

        dummy_input = torch.zeros((1, 3, 640, 480))
        torch.onnx.export(model, dummy_input, SavePath[idx],opset_version=11)

        omodel = onnx.load(SavePath[idx])
        onnx.checker.check_model(omodel)
