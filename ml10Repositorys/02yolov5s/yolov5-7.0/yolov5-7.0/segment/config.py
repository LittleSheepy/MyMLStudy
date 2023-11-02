# -*- coding: utf-8 -*-
import argparse
import math

#出错原因：git环境变量设置问题
#简便解决办法：在导入包的上方增加以下代码
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

'''
print(ROOT)         # ..
print(Path.cwd())   # 文件所在的根目录
'''


class ConfigurationParameter:

    '''
    Train:
    python segment/train.py
                --weights weights/yolov5m-seg.pt
                --cfg models/segment/yolov5m-seg-cmps.yaml
                --data data/coco128-seg-cmps.yaml
                --epochs 300
                --img 640
                --batch-size 8
                --device 0
                --workers
                --project
    '''

    trian_weights = ROOT / r"weights/yolov5s-seg.pt"
    train_cfg = ROOT / r"models/segment/yolov5s-seg.yaml"
    train_data = ROOT / r"data/coco128-seg12_cm_ps.yaml"
    train_epochs = 300
    train_batch_size = 8
    train_img_size = 1024
    train_device = 0
    train_workers = 8
    train_project = ROOT / 'runs/12train-seg-cmps'

    '''
    Predict:
    python segment/predict.py 
                --source ??? 测试图片路径
                --data data/coco128-seg-cmps.yaml
                --weights runs\\train-seg\\exp\\weights\\best.pt 
                --device 0   使用GPU(0,1,2)
    '''
    predict_source = r""
    predict_data = ROOT / r"data/coco128-seg-cmps.yaml"
    predict_wights = ROOT / R"runs\\train-seg\\exp\\weights\\best.pt"
    predict_device = 0







