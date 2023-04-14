import tensorflow as tf
import numpy as np
from lib.net.DarkNet19 import DarkNet19
from yolo.YOLOv3 import YOLOv3

print(tf.__version__)

class WeightReader():
    def __init__(self, weightPath, net):
        self.net = net
        self.weightPath = weightPath
        self.all_weights = np.fromfile(weightPath, dtype='float32')
        self.offset = 4

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4
    def load_weights(self):
        weights = self.all_weights
        var_list = tf.global_variables()
        ptr = self.offset
        i = 0
        assign_ops = []
        while i < len(var_list) - 1:
            # detector/darknet-53/Conv/BatchNorm/
            var1 = var_list[i]
            var2 = var_list[i + 1]
            # do something only if we process conv layer
            if 'conv' in var1.name.split('/')[-2]:
                # check type of next layer,BatchNorm param first of weight
                if 'bn' in var2.name.split('/')[-2]:
                    # load batch norm params, It's equal to l.biases,l.scales,l.rolling_mean,l.rolling_variance
                    gamma, beta, mean, var = var_list[i + 1:i + 5]
                    batch_norm_vars = [beta, gamma, mean, var]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        num_params = np.prod(shape)
                        var_weights = weights[ptr:ptr + num_params].reshape(shape)
                        ptr += num_params
                        print(var, ptr)
                        assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                    # we move the pointer by 4, because we loaded 4 variables
                    i += 4
                elif 'conv' in var2.name.split('/')[-2]:
                    # load biases,not use the batch norm,So just only load biases
                    bias = var2
                    bias_shape = bias.shape.as_list()
                    bias_params = np.prod(bias_shape)
                    bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                    ptr += bias_params
                    print(bias, ptr)
                    assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                    # we loaded 1 variable
                    i += 1

                # we can load weights of conv layer
                shape = var1.shape.as_list()
                num_params = np.prod(shape)

                var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                # remember to transpose to column-major
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                print(var1, ptr)
                assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
                i += 1
            # yolov3中不需要以下部分
            elif 'Local' in var1.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                print(bias, ptr)
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                i += 1

                # we can load weights of conv layer
                shape = var1.shape.as_list()
                num_params = np.prod(shape)

                var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                # remember to transpose to column-major
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                print(var1, ptr)
                assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
                i += 1
            elif 'Fc' in var1.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                print(bias, ptr)
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                i += 1

                # we can load weights of conv layer
                shape = var1.shape.as_list()
                num_params = np.prod(shape)

                var_weights = weights[ptr:ptr + num_params].reshape(shape[1], shape[0])
                # remember to transpose to column-major
                var_weights = np.transpose(var_weights, (1, 0))
                ptr += num_params
                print(var1, ptr)
                assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
                i += 1
            # elif "L2Normalization" in var1.name.split('/')[-2]:

        return assign_ops





if __name__ == '__main__':
    weightPath_voc = "D:\ML_datas\weights\YOLOs\YOLO_v2/yolov2-voc.weights"
    weightPath_coco = "D:\ML_datas\weights\YOLOs\YOLO_v2/yolov2.weights"    # 50983565
    weightPath_v3 = "D:\ML_datas\weights\YOLOs\YOLO_v3/yolov3.weights"

    saveweightspath_v3 = "D:\ML_datas\weights\YOLOs\YOLO_v3\YOLOS-tensorflow\yolov3-coco.ckpt"
    saveweightspath = "D:\ML_datas\weights\YOLOs\YOLO_v2\dk-yolov2-coco\dk-yolov2-coco.ckpt"

    weightPath = weightPath_coco
    net = DarkNet19()
    var_list = tf.global_variables()
    variable_to_restore = tf.global_variables(scope="DarkNet19")
    weightReader = WeightReader(weightPath,net)
    assign_ops = weightReader.load_weights()
    sess = tf.Session()
    sess.run(assign_ops)
    saver=tf.train.Saver()
    saver.save(sess,saveweightspath)

    # # Read data from checkpoint file
    # from tensorflow.python import pywrap_tensorflow
    # reader = pywrap_tensorflow.NewCheckpointReader(saveweightspath)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor_name: ", key)





