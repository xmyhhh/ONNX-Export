import torch as nn

from Pytorch.YoloV5.models.yolo import Model
from Pytorch.YoloV5.models.experimental import attempt_load
def YoloV5ModelBuilder():
    return attempt_load(weights='yolov5s.pt',device='cpu')




