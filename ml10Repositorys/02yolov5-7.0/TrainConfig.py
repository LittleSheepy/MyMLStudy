# -*- coding: utf-8 -*-
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import time

class ConfigurationParameter:
    '''
    Train:
    python segment/train.py
                --weights weights/yolov5m-seg.pt
                --cfg models/segment/yolov5m-seg-LGCB.yaml
                --data data/12-seg-CMPS.yaml
                --epochs 300
                --img 640
                --batch-size 8
                --device 0
                --workers
                --project
    '''
    # train_weights = r"weights/yolov5s-seg.pt"  # 预训练权重
    train_weights = r"D:\08weight\05yolov5\7.0\seg/yolov5s-seg.pt"  # 预训练权重
    train_cfg = r"models/segment/yolov5s-seg.yaml"  # cfg
    # train_data = r"data/12_Side_CMPS.yaml"  # 训练数据配置文件
    train_data = r"data/12LunTai/LunTai.yaml"  # 训练数据配置文件
    train_epochs = 1000  # 训练批次
    train_batch_size = 20  # batch—size
    train_img_size = 1024  # 图片大小[Front用1024，Side用1024，Back用2048]
    train_device = 0  # GPU
    train_workers = 4
    train_project = r'runs_liu\train-seg/'  # 保存位置
    train_name = "exp_luntai_" + time.strftime('%Y%m%d', time.localtime())
    train_optimizer = "SGD"           # choices=['SGD', 'Adam', 'AdamW']默认SGD
    '''
    data/01_Front_LZPS.yaml
    data/05_Front_JY.yaml
    data/08_Back_DMPS.yaml
    data/09_Back_DMLB.yaml
    data/11_Back_SYJ.yaml
    data/16_Front_YW.yaml
    '''