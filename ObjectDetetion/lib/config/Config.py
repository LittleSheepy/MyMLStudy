import os
import threading
def read_coco_labels(file_path):   #读取coco数据集类别
    f = open(file_path)
    class_names = []
    for l in f.readlines():
        l = l.strip()     #去掉字符串左右两遍的空格
        class_names.append(l)
    return class_names

"""
配置说明：
SSD：
数据集voc
YOLOv3:
数据集：coco
"""

class Config(object):
    print("Config")
    # data
    data_path_base = "D:/ML_datas/"
    cache_file = data_path_base + "\VOCdevkit\VOC2007/train\pascalVocDS_train.pkl"
    coco_class_names = read_coco_labels(data_path_base + "coco/coco_classes.txt")  # 返回类别
    anchors_coco = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]
    anchors_voc = [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    data_set_name = "voc"
    data_set_info = {
        "voc" :{"num_class":20, "name_class":voc_classes, "n_output_channels":125, "anchors":anchors_voc,
                "data_path":data_path_base + "/VOCdevkit/VOC2007/train/",
                "imgNameListFile":"trainval.txt"},
        "coco" :{"num_class":80, "name_class":coco_class_names, "n_output_channels":425, "anchors":anchors_coco,
                 "data_path":data_path_base + "\coco\coco2014/"}
    }
    ds = data_set_info[data_set_name]
    data_path = data_set_info[data_set_name]["data_path"]
    anchors = data_set_info[data_set_name]["anchors"]
    num_class = data_set_info[data_set_name]["num_class"]
    class_names = data_set_info[data_set_name]["name_class"]
    # train
    phase = "train"
    training = True
    dropout_keep_prob = 0.5
    image_size = 416
    max_iter = 100000
    saver_iter = 100
    lr = 0.0001
    batch_size = 1
    flipped = True
    # original_model = data_path_base + "weights\SSD\ssd_checkpoints\ssd_vgg_300_weights.ckpt"
    original_model = data_path_base + "weights\SSD\SSD-Tensorflow\ssd_300_vgg.ckpt"
    data_test_img = data_path_base + "testImage/944.jpg"

    # 单列模式
    def __new__(self, *args, **kwargs):
        self._instance_lock = threading.Lock()
        if not hasattr(self, "_instance"):
            with self._instance_lock:
                if not hasattr(self, "_instance"):
                    self._instance = object.__new__(self)
        return self._instance

class ConfigYOLOv2(Config):
    # train
    image_size = 416
    cell_size = 13
    box_per_cell = 5
    n_output_channels = Config.data_set_info[Config.data_set_name]["n_output_channels"]
    data_set_info = {
        "voc" :{
            "per_model_restore_path":"D:\ML_datas\weights\YOLOs\YOLO_v2\dk-yolov2-voc/dk-yolov2-voc.ckpt"
        },
        "coco" :{
            "per_model_restore_path":"D:\ML_datas\weights\YOLOs\YOLO_v2\dk-yolov2-coco/dk-yolov2-coco.ckpt"
        }
    }
    # per_model_restore_path = "D:\ML_datas\weights\YOLOs\YOLO_v2\darknet19\model/yolo2_coco.ckpt"
    # per_model_restore_path = "D:\ML_datas\weights\YOLOs\YOLO_v2\dk-yolov2-coco/dk-yolov2-coco.ckpt"
    per_model_restore_path = data_set_info[Config.data_set_name]["per_model_restore_path"]
    # per_model_restore_path = "D:\ML_datas\weights\my_weights\YOLOs_tensorflow/yolo_v2_voc.ckpt"

    # save
    logs = "D:\logs/yolov2/"
    # data
    pascal_path = os.path.join(Config.data_path_base, 'pascal_voc')
    cache_path = os.path.join(Config.data_path_base, 'processedData', "YOLOs_tensorflow")
    OUTPUT_DIR = os.path.join(pascal_path, 'output')
    weight_path = "D:\ML_datas\weights\my_weights/YOLOs_tensorflow/"


class ConfigYOLOv3(Config):
    mode_name = "YOLOv3"
    anchor_per_scale = 3
    iou_loss_thresh = 0.5
    strides = [8, 16, 32]
    original_model = Config.data_path_base + "weights\YOLOs\YOLO_v3\YOLOS-tensorflow\yolov3-coco.ckpt"
    # data
    anchors = [[[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],
               [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],
               [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]]
    input_sizes = [416]  # [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
    test_image_size = 416
    annot_path = Config.data_path_base + "YOLOs\data\dataset/voc_train.txt"

    cache_file = Config.data_path + Config.data_set_name + mode_name + "pascalVocDS_train.pkl"
    # train
    upsample_method = "resize"
    warmup_epochs = 2
    steps_per_epochs = 10000
    first_stage_epochs = 20
    second_stage_epochs = 30
    lr_init = 1e-4
    lr_end = 1e-6
    moving_ave_decay = 0.9995
    data = Config.data_path_base
    logs = "D:\logs/yolov3/"

class SSD300(Config):
    mode_name = "SSD300"
    feature_layers = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
    feature_shapes = [38, 19, 10, 5, 3, 1]
    strides = [8, 16, 32, 64, 100, 300]
    image_size = 300
    anchor_sizes_num = 2
    anchor_sizes = [(21., 45.),(45., 99.),(99., 153.),(153., 207.),(207., 261.),(261., 315.)]
    anchor_ratios = [[2, .5],[2, .5, 3, 1. / 3], [2, .5, 3, 1. / 3], [2, .5, 3, 1. / 3], [2, .5], [2, .5]]
    num_anchors = [len(s)+len(r) for s, r in zip(anchor_sizes, anchor_ratios)]          # [anchor_sizes_num*len(anchor_ratios[i]) for i in range(6)] 报错
    num_class = Config.data_set_info[Config.data_set_name]["num_class"]+1
    # data

    cache_file = Config.data_path + Config.data_set_name + "_" + mode_name + ".pkl"
    # saver
    logdir = "D:/logs/ssd"



config = SSD300()
