import tensorflow as tf
import numpy as np

from lib.config.Config import ConfigYOLOv2
from lib.tools.utils import *
import lib.tools.utils

config = ConfigYOLOv2()
class DarkNet19():
    def __init__(self):
        with tf.variable_scope("DarkNet19"):
            self.output = self.build_network()

    def build_network(self):
        self.netlist = {}
        self.images = tf.placeholder(tf.float32, [None, config.image_size, config.image_size, 3], name='images')    # 假设[-1, 224, 224, 3].主要是和论文对应
        net = conv2d(self.images, 32, 3, 1, name='conv1')  # 卷积层，卷积核数量32，大小为3*3，padding=1, 默认步长为1
        net = maxpool(net, size=2, stride=2, name='pool1')  # maxpooling, 减少特征图的维度一半，为112*112,因为siez=2*2,步长为2
        self.netlist["1"] = net
        net = conv2d(net, 64, 3, 1, name='conv2')  # 卷积层，卷积核数量为64，大小为3*3，padding=1,默认步长为1
        net = maxpool(net, 2, 2, name='pool2')  # maxpooling，变成56*56
        self.netlist["2"] = net

        net = conv2d(net, 128, 3, 1, name='conv3_1')  # 卷积层，卷积核数量为128，大小为3*3，padding=1,默认步长为1
        net = conv2d(net, 64, 1, 0, name='conv3_2')  # 卷积层，卷积核数量为64，大小为1*1，padding=0,默认步长为1
        net = conv2d(net, 128, 3, 1, name='conv3_3')  # 卷积层，卷积核数量为128，大小为3*3，padding为1，默认步长为1
        net = maxpool(net, 2, 2, name='pool3')  # maxpooling,变成28*28
        self.netlist["3"] = net

        net = conv2d(net, 256, 3, 1, name='conv4_1')  # 卷积层，卷积核数量为256，大小为3*3，padding=1,默认步长为1
        net = conv2d(net, 128, 1, 0, name='conv4_2')  # 卷积层，卷积核数量为128，大小为1*1，padding=0，默认步长为1
        net = conv2d(net, 256, 3, 1, name='conv4_3')  # 卷积层，卷积核数量为256，大小为3*3，padding=1,默认步长为1
        net = maxpool(net, 2, 2, name='pool4')  # maxpooling,变成14*14
        self.netlist["4"] = net

        net = conv2d(net, 512, 3, 1, name='conv5_1')  # 卷积层，卷积核数量为512，大小为3*3，padding=1,默认步长为1
        net = conv2d(net, 256, 1, 0, name='conv5_2')  # 卷积层，卷积核数量为256，大小为1*1，padding=0,默认步长为1
        net = conv2d(net, 512, 3, 1, name='conv5_3')  # 卷积层，卷积核数量为512，大小为3*3，padding=1,默认步长为1
        net = conv2d(net, 256, 1, 0, name='conv5_4')  # 卷积层，卷积核数量为256，大小为1*1，padding=0,默认步长为1
        shortcut = conv2d(net, 512, 3, 1, name='conv5_5')  # [-1,14,14,512] 存储这一层特征图，以便后面passthrough层
        net = maxpool(shortcut, 2, 2, name='pool5')  # maxpooling，变成7*7
        self.netlist["5"] = net

        net = conv2d(net, 1024, 3, 1, name='conv6_1')  # 卷积层，卷积核数量为1024,大小为3*3，padding=1,默认步长为1
        net = conv2d(net, 512, 1, 0, name='conv6_2')  # 卷积层，卷积核数量为512，大小为1*1，padding=0,默认步长为1
        net = conv2d(net, 1024, 3, 1, name='conv6_3')  # 卷积层，卷积核数量为1024，大小为3*3，padding=1，默认步长为1
        net = conv2d(net, 512, 1, 0, name='conv6_4')  # 卷积层，卷积核数量为512，大小为1*1，padding=0,默认步长为1
        net = conv2d(net, 1024, 3, 1, name='conv6_5')  # 卷积层，卷积核数量为1024，大小为3*3，padding=1,默认步长为1
        self.netlist["6"] = net

        # 具体这个可以参考： https://blog.csdn.net/hrsstudy/article/details/70767950     Training for classification 和 Training for detection
        # 训练检测网络时去掉了分类网络的网络最后一个卷积层，在后面增加了三个卷积核尺寸为3 * 3，卷积核数量为1024的卷积层，并在这三个卷积层的最后一层后面跟一个卷积核尺寸为1 * 1
        # 的卷积层，卷积核数量是（B * （5 + C））。
        # 对于VOC数据集，卷积层输入图像尺寸为416 * 416
        # 时最终输出是13 * 13
        # 个栅格，每个栅格预测5种boxes大小，每个box包含5个坐标值和20个条件类别概率，所以输出维度是13 * 13 * 5 * （5 + 20）= 13 * 13 * 125。
        #
        # 检测网络加入了passthrough
        # layer，从最后一个输出为26 * 26 * 512
        # 的卷积层连接到新加入的三个卷积核尺寸为3 * 3
        # 的卷积层的第二层，使模型有了细粒度特征。

        # 下面这部分主要是training for detection
        net = conv2d(net, 1024, 3, 1, name='conv7_1')  # 卷积层，卷积核数量为1024，大小为3*3,padding=1,默认步长为1
        net = conv2d(net, 1024, 3, 1, name='conv7_2')  # 卷积层，卷积核数量为1024，大小为3*3，padding=1,默认步长为1，大小为1024*7*7
        self.netlist["7"] = net

        # 关于这部分细粒度的特征的解释，可以参考：https://blog.csdn.net/hai_xiao_tian/article/details/80472419
        # shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行passthrough处理
        # 这样26*26*512 -> 26*26*64 -> 13*13*256的特征图，可能是输入图片刚开始不是224，而是448，知道就好了,YOLOv2的输入图片大小为 416*416
        shortcut = conv2d(shortcut, 64, 1, 0, name='conv_shortcut')  # 卷积层，卷积核数量为64，大小为1*1，padding=0,默认步长为1，变成26*26*64
        shortcut = reorg(shortcut, 2)  # passthrough, 进行Fine-Grained Features，得到13*13*256
        # 连接之后，变成13*13*（1024+256）
        net = tf.concat([shortcut, net],
                        axis=-1)  # channel整合到一起，concatenated with the original features，passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上，
        net = conv2d(net, 1024, 3, 1, name='conv8')  # 卷积层，卷积核数量为1024，大小为3*3，padding=1, 在连接的特征图的基础上做卷积进行预测。变成13*13*1024

        # detection layer: 最后用一个1*1卷积去调整channel，该层没有BN层和激活函数，变成: S*S*(B*(5+C))，在这里为：13*13*425
        output = conv2d(net, filters_num=config.n_output_channels, filters_size=1, batch_normalize=False, activation=None,
                        use_bias=True, name='conv_dec')
        return output

    def decode_output(self):
        '''
         model_output:darknet19网络输出的特征图
         output_sizes:darknet19网络输出的特征图大小，默认是13*13(默认输入416*416，下采样32)
        '''
        model_output = self.output
        output_sizes = (13, 13)
        num_class = config.num_class
        anchors = config.anchors
        H, W = output_sizes  # H=13, W=13
        num_anchors = len(anchors)  # num_anchors=5
        anchors = tf.constant(anchors, dtype=tf.float32)  # 将anchors转变成tf格式的常量列表

        # 13*13*num_anchors*(num_class+5)，第一个维度自适应batchsize
        detection_result = tf.reshape(model_output, [-1, H * W, num_anchors, num_class + 5])  # 注意reshape的用法

        # darknet19网络输出转化——偏移量、置信度、类别概率
        xy_offset = tf.nn.sigmoid(detection_result[:, :, :, 0:2])  # 中心坐标相对于该cell坐上角的偏移量，用sigmoid函数归一化到(0,1)
        wh_offset = tf.exp(detection_result[:, :, :, 2:4])  # 相对于特征图的wh比例，通过e指数解码
        obj_probs = tf.nn.sigmoid(detection_result[:, :, :, 4])  # 置信度，sigmoid函数归一化到(0,1)
        class_probs = tf.nn.softmax(detection_result[:, :, :, 5:])  # 网络回归的是得分，用softmax转变成概率，使其值落在(0,1)

        # 构建特征图每个cell的左上角的xy坐标
        height_index = tf.range(H, dtype=tf.float32)  # range(0,13)
        width_index = tf.range(W, dtype=tf.float32)  # range(0,13)
        # 变成x_cell=[[0,1,...,12],...,[0,1,...,12]]和y_cell=[[0,0,...,0],[1,...,1]...,[12,...,12]]
        x_cell, y_cell = tf.meshgrid(height_index, width_index)
        x_cell = tf.reshape(x_cell, [1, -1, 1])  # 和上面[H*W,num_anchors,num_class+5]对应
        y_cell = tf.reshape(y_cell, [1, -1, 1])

        # decode
        bbox_x = (x_cell + xy_offset[:, :, :, 0]) / W
        bbox_y = (y_cell + xy_offset[:, :, :, 1]) / H
        bbox_w = (anchors[:, 0] * wh_offset[:, :, :, 0]) / W  # 先验框宽度*偏移值
        bbox_h = (anchors[:, 1] * wh_offset[:, :, :, 1]) / H  # 先验框高度*偏移值
        # 中心坐标+宽高box(x,y,w,h) -> xmin=x-w/2 -> 左上+右下box(xmin,ymin,xmax,ymax)
        bboxes = tf.stack([bbox_x - bbox_w / 2, bbox_y - bbox_h / 2, bbox_x + bbox_w / 2, bbox_y + bbox_h / 2],
                          axis=3)  # 转变成坐上-右下坐标

        return bboxes, obj_probs, class_probs  # 这个返回框的坐标（左上角-右下角），目标置信度，类别概率


print(config.image_size)
test_utils()