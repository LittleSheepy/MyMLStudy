import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import numpy as np
import cv2

import Framework as myf
from BufferDS import BufferDS
from Celeba import Celeba

class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.sample_path = 'MNIST_data'
        self.vector_size = 4
        self.momentum = 0.99
        self.cols = 20
        self.img_path = 'imgs/{name}/test.jpg'.format(name=self.get_name())

        self.buffer_size = 3
        self.batch_size = 200
        self.lr = 0.0001
        self.persons = 0
        self.base_filters = 32
        self.img_size = 32 * 4
        self.convs = 5
        self.is_train = True
        # self.epoches = 4

    def get_name(self):
        return 'LiuNetWork'

    def get_sub_tensors(self, gpu_index):
        return MySubTensors(self)

    def get_tensors(self):
        return MyTensors(self)
    def get_da_train(self):
        return self.ds
    def get_da_test(self):
        return self.ds

class MyTensors(myf.Tensors):
    def get_loss_for_summary(self, loss):
        return tf.sqrt(loss)


class MySubTensors:
    def __init__(self, config: MyConfig):
        self.config = config
        classes = 10177
        with tf.device('/gpu:0'):
            x = tf.placeholder(tf.float32, [None, 218, 178, 3], 'x')
            self.y = tf.placeholder(tf.int32, [None], 'y')
            self.inputs = [x, self.y]

            # conv1
            with tf.variable_scope("conv1") as scope:
                x = tf.layers.conv2d(x, 64, 7, strides=2, padding="same", name='conv1')        # (?,109,89,64)
                x = tf.layers.batch_normalization(x, training=config.is_train, name='conv1_bn')
                x = tf.nn.relu(x, name="relu")
                x = tf.layers.max_pooling2d(x, 3, 2, "same")# (?,55,45,64)
            #conv2
            #x = self.residual_block(x, 64, 1, name="conv2_1")       # (?,55,45,64)

            x = self.LiuNetWork(x, False, "LiuNetWork")
            #x = self.residual_block(x, 64, 1, name='conv2_2')
            #conv3
            x = self.residual_block(x, 128, 2, name="conv3_1")      # (?,28,23,128)
            #x = self.residual_block(x, 128, 1, name='conv3_2')      # (?,28,23,128)
            # x = self.LiuNetWork(x, False, "LiuNetWork")
            # conv4
            x = self.residual_block(x, 256, 2, name="conv4_1")      # (?,14,12,256)
            #x = self.residual_block(x, 256, 1, name='conv4_2')      # (?,14,12,256)
            # conv5
            x = self.residual_block(x, 512, 2, name="conv5_1")      # (?,7,6,512)
            #x = self.residual_block(x, 512, 1, name='conv5_2')      # (?,7,6,512)
            # conv6
            #x = self.residual_block(x, 512, 2, name="conv6_1")      # (?,4,3,512)
            #x = self.residual_block(x, 512, 1, name='conv5_2')      # (?,4,3,512)

            # x = tf.layers.flatten(x)  # (?, 6144)
            x = tf.reduce_mean(x, [1, 2])  # (?,512)

            x = tf.layers.dense(x, 10177)

            self.logits = x
            self.y_hot = tf.one_hot(self.y, classes)
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_hot, logits=self.logits)
            self.losses = [tf.reduce_mean(self.loss)]
            # self.loss_summary = tf.summary.scalar("loss", self.losses)



            self.y_pre = tf.argmax(self.logits, axis=1, output_type=tf.int32)
            self.y = tf.argmax(self.y_hot, axis=1, output_type=tf.int32)

            self.precise = tf.cast(tf.equal(self.y, self.y_pre),tf.float32)
            self.precise = tf.reduce_mean(self.precise)
            self.precise_summary = tf.summary.scalar("precise", self.precise)

            self.lr = tf.placeholder(tf.float32, shape=[], name="lr")
            # opt = tf.train.MomentumOptimizer(self.lr, 0.9)
            opt = tf.train.AdamOptimizer(self.lr)
            # opt = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = opt.minimize(self.loss)

    def LiuNetWork(self,x, Endflg=False, name="LiuNetWork"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            if Endflg:
                x = tf.layers.conv2d(x, num_channel, 3, strides=1, padding="same", name='conv_1')
                x = tf.layers.batch_normalization(x, training=self.config.is_train, name='bn_1')
                x = tf.nn.relu(x, name='relu_1')

                x = tf.layers.conv2d(x, num_channel, 3, strides=1, padding="same", name='conv_2')
                x = tf.layers.batch_normalization(x, training=self.config.is_train, name='bn_2')
                x = tf.nn.relu(x, name='relu_2')
                return x
            else:
                delt = self.LiuNetWork(x, Endflg=True)
                x = 0.9*x + 0.1*delt
                return self.LiuNetWork(x, Endflg=True)



    def residual_block(self, x, out_channel, strides, name="unit"):  # 128 3
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            if in_channel == out_channel and strides == 1:
                shortcut = tf.identity(x)
                # shortcut = x
            else:
                shortcut = tf.layers.conv2d(x, out_channel, 3, strides=strides, padding="same", name='shortcut')  # (?,_,_,out_channel)
            # Residual
            x = tf.layers.conv2d(x, out_channel, 3, strides=strides, padding="same", name='conv_1')
            x = tf.layers.batch_normalization(x, training=self.config.is_train, name='bn_1')
            x = tf.nn.relu(x, name='relu_1')
            x = tf.layers.conv2d(x, out_channel, 3, strides=1, padding="same", name='conv_2')
            x = tf.layers.batch_normalization(x, training=self.config.is_train, name='bn_2')
            # Merge
            x = x + shortcut
            x = tf.nn.relu(x, name='relu_2')
        return x

if __name__ == '__main__':
    path = "D:/BaiduYunDownload/15_Ce.leb.Faces Att.ribu.tes Da.ta.set (CelebA)/Img/img_align_celeba.zip"
    path_anno = "D:/BaiduYunDownload/15_Ce.leb.Faces Att.ribu.tes Da.ta.set (CelebA)/Anno/identity_CelebA.txt"
    path_bbox = "D:/BaiduYunDownload/15_Ce.leb.Faces Att.ribu.tes Da.ta.set (CelebA)/Anno/list_bbox_celeba.txt"
    celeba = Celeba(path, path_anno, path_bbox)
    cfg = MyConfig()
    ds = BufferDS(cfg.buffer_size, celeba, cfg.batch_size)
    cfg.persons = celeba.persons
    cfg.ds = ds
    cfg.from_cmd()



