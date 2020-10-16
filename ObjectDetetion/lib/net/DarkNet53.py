import tensorflow as tf
import numpy as np

from lib.config.Config import ConfigYOLOv2
from lib.tools.utils import *
import lib.tools.utils

config = ConfigYOLOv2()

class DarkNet53():
    def __init__(self, images):
        with tf.variable_scope("DarkNet53"):
            self.build_network(images)

    def build_network(self, images):
        net = conv2d(images, 32, 3, 1, name="conv1")
        net = conv2d(net, 64, 3, 1, 2, name="conv2")
        net = self.residual_blocks(1, net, 32, 64, 0)
        net = conv2d(net, 128, 3, 1, 2, name="conv5")
        net = self.residual_blocks(2, net, 64, 128, 1)
        net = conv2d(net, 256, 3, 1, 2, name="conv10")
        net = self.residual_blocks(8, net, 128, 256, 3)
        self.route_1 = net
        net = conv2d(net, 512, 3, 1, 2, name="conv27")
        net = self.residual_blocks(8, net, 256, 512, 11)
        self.route_2 = net
        net = conv2d(net, 1024, 3, 1, 2, name="conv44")
        self.output = self.residual_blocks(4, net, 512, 1024, 19)

    def residual_blocks(self, res_batch, input_data, filter_num1, filter_num2, num):
        for i in range(res_batch):
            input_data = residual_block(input_data, filter_num1, filter_num2, name='residual%d' % (i + num))
        return input_data


if __name__ == '__main__':
    dk53 = DarkNet53()
