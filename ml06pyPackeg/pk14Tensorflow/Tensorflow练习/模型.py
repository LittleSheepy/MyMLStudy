import numpy as np
import tensorflow as tf
import pprint # 使用pprint 提高打印的可读性
print(tf.__version__)

ckpt_path = "D:\ML_datas\weights\SSD\ssd_checkpoints\ssd_vgg_300_weights.ckpt"
meta_path = "D:\ML_datas\weights\SSD\ssd_checkpoints\ssd_vgg_300_weights.ckpt.meta"
ckpt_path = "D:\ML_datas\weights\SSD\SSD-Tensorflow\ssd_300_vgg.ckpt"
yolov3 = "D:\ML_datas\weights\YOLOs\YOLO_v3\YOLOS-tensorflow\yolov3-coco.ckpt"
yolov3_meta = "D:\ML_datas\weights\YOLOs\YOLO_v3\YOLOS-tensorflow\yolov3-coco.ckpt.meta"
ckpt_path = yolov3
meta_path = yolov3_meta
saver = tf.train.import_meta_graph(meta_path)
variables = tf.trainable_variables()
for v in variables:
    print(v.name, v.shape)

print("*"*50)
from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
print("debug_string:\n")
pprint.pprint(reader.debug_string().decode("utf-8"))
print("*"*50)
var_to_shape_map = reader.get_variable_to_shape_map()
#tensor =  reader.get_tensor("global_step")
print("len ",len(var_to_shape_map))
pprint.pprint(var_to_shape_map)