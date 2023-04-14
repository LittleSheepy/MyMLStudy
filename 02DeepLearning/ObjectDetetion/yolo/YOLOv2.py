import tensorflow as tf
import numpy as np
import cv2

from lib.net.DarkNet19 import DarkNet19
from lib.config.Config import config as config
from lib.config.Config import ConfigYOLOv2
from lib.datasets.pascal_voc import pascal_voc
from lib.tools.utils import *
import lib.tools.utils


class YOLOv2():
    object_scale = 5.0
    noobject_scale = 1.0
    class_scale = 1.0
    coordinate_scale = 1.0

    def __init__(self):
        self.batch_size = config.batch_size
        self.cell_size = config.cell_size
        self.box_per_cell = config.box_per_cell
        self.anchor = [0.57273, 1.87446, 3.33843, 7.88282, 9.77052, 0.677385, 2.06253, 5.47434, 3.52778, 9.16828]
        self.num_class = config.num_class
        self.initial_learn_rate = 0.0001
        self.saver_iter = 100
        # self.offset = np.transpose(np.reshape(np.array(
        #     [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
        #     (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.offset1 = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
                                         [self.box_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        self.offset = tf.reshape(tf.constant(self.offset1, dtype=tf.float32), [1, self.cell_size, self.cell_size, self.box_per_cell])
        self.offset = tf.tile(self.offset, (self.batch_size, 1, 1, 1))
        with tf.device('/gpu:0'):
            self.net = DarkNet19()

    def cal_loss(self):
        self.net.labels = tf.placeholder(tf.float32, [None, config.cell_size, config.cell_size, 1, config.num_class + 5], name = 'labels')  # (?, 13, 13, 1, 25)
        predict = self.net.output   # (?, 13, 13, 425)  (?, 13, 13, 125=5*25)
        label = tf.tile(self.net.labels, (1, 1, 1, 5, 1))       # (?, 13, 13, 5, 25)
        predict = tf.reshape(predict, [config.batch_size, config.cell_size, config.cell_size, config.box_per_cell, config.num_class + 5])   # (?, 13, 13, 425)  (?, 13, 13, 125=5*25)
        box_coordinate = tf.reshape(predict[:, :, :, :, :4], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])       # 坐标 (1, 13, 13, 5, 4)
        box_confidence = tf.reshape(predict[:, :, :, :, 4], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 1])        # 目标置信度 (1, 13, 13, 5, 1)
        box_classes = tf.reshape(predict[:, :, :, :, 5:], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class])  # 类别 (1, 13, 13, 5, 20)

        boxes1 = tf.stack([(1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 0])) + self.offset) / self.cell_size,
                           (1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 1])) + tf.transpose(self.offset, (0, 2, 1, 3))) / self.cell_size,
                           tf.sqrt(tf.exp(box_coordinate[:, :, :, :, 2]) * np.reshape(self.anchor[:5], [1, 1, 1, 5]) / self.cell_size),
                           tf.sqrt(tf.exp(box_coordinate[:, :, :, :, 3]) * np.reshape(self.anchor[5:], [1, 1, 1, 5]) / self.cell_size)])
        box_coor_trans = tf.transpose(boxes1, (1, 2, 3, 4, 0))
        box_confidence = 1.0 / (1.0 + tf.exp(-1.0 * box_confidence))
        box_classes = tf.nn.softmax(box_classes)

        response = tf.reshape(label[:, :, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell])
        boxes = tf.reshape(label[:, :, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])
        classes = tf.reshape(label[:, :, :, :, 5:], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class])

        iou = self.calc_iou(box_coor_trans, boxes)
        best_box = tf.to_float(tf.equal(iou, tf.reduce_max(iou, axis=-1, keep_dims=True)))
        confs = tf.expand_dims(best_box * response, axis = 4)

        conid = self.noobject_scale * (1.0 - confs) + self.object_scale * confs
        cooid = self.coordinate_scale * confs
        proid = self.class_scale * confs

        coo_loss = cooid * tf.square(box_coor_trans - boxes)
        con_loss = conid * tf.square(box_confidence - confs)
        pro_loss = proid * tf.square(box_classes - classes)

        loss = tf.concat([coo_loss, con_loss, pro_loss], axis = 4)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis = [1, 2, 3, 4]), name = 'loss')
        self.losses = {}

        self.losses["coo_loss"] = coo_loss
        self.losses["con_loss"] = con_loss
        self.losses["pro_loss"] = pro_loss
        self.losses["loss"] = loss
        return loss

    def train(self):
        self.variable_to_restore = tf.global_variables(scope="DarkNet19")
        with tf.device('/gpu:0'):
            self.loss = self.cal_loss()
            # train_op
            #self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            # self.learn_rate = tf.train.exponential_decay(self.initial_learn_rate, self.global_step, 20000, 0.1, name='learn_rate')
            self.learn_rate = self.initial_learn_rate
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss) # , global_step=self.global_step

            self.average_op = tf.train.ExponentialMovingAverage(0.999).apply(tf.trainable_variables())
            with tf.control_dependencies([self.optimizer]):
                self.train_op = tf.group(self.average_op)

        self.data_train = pascal_voc()
        self.loss_summary = tf.summary.scalar('loss', self.loss)

        self.summary_op = tf.summary.merge_all()
        cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=cfg)
        self.sess.run(tf.global_variables_initializer())


        self.saver = tf.train.Saver(self.variable_to_restore)
        print('Restore weights from:', config.per_model_restore_path)
        self.saver.restore(self.sess, config.per_model_restore_path)
        self.writer = tf.summary.FileWriter(config.logs, self.sess.graph)

        for step in range(config.max_iter):
            images, labels = self.data_train.next_batches()
            feed_dict = {self.net.images: images, self.net.labels: labels}
            loss_summary,losses,layer_value, _ = self.sess.run([self.loss_summary,self.losses, self.net.netlist,self.train_op], feed_dict=feed_dict)
            if step % 50 == 0:
                self.writer.add_summary(loss_summary, step)
            if step % self.saver_iter == 0:
                self.saver.save(self.sess, config.weight_path + '/yolo_v2_voc.ckpt')

    def test_image(self, img_file=None):
        if not img_file:
            img_file = config.data_test_img

        image = cv2.imread(img_file)
        image_cp = preprocess_image(image, process="resize")  #图像预处理，resize image, normalization归一化， 增加一个在第0的维度--batch_size
        output_decoded = self.net.decode_output()
        cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=cfg) as sess:
            sess.run(tf.global_variables_initializer())  # 初始化tensorflow全局变量
            saver = tf.train.Saver()
            saver.restore(sess, config.per_model_restore_path)  # 把模型加载到当前session中
            bboxes, obj_probs, class_probs = sess.run(output_decoded,
                                                      feed_dict={self.net.images: image_cp})  # 这个函数返回框的坐标，目标置信度，类别置信度

        bboxes, scores, class_max_index = postprocess(bboxes, obj_probs, class_probs,
                                                      image_shape=image.shape[:2])  # 得到候选框之后的处理，先留下阈值大于0.5的框，然后再放入非极大值抑制中去
        colors = generate_colors(config.class_names)
        img_detection = draw_detection(image, bboxes, scores, class_max_index, config.class_names, colors)  # 得到图片

        cv2.imshow("detection_results", img_detection)  # 显示图片
        cv2.waitKey(0)  # 等待按任意键退出

    def calc_iou(self, boxes1, boxes2):
        boxx = tf.square(boxes1[:, :, :, :, 2:4])
        boxes1_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tf.stack([boxes1[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes1[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes1 = tf.transpose(box, (1, 2, 3, 4, 0))

        boxx = tf.square(boxes2[:, :, :, :, 2:4])
        boxes2_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tf.stack([boxes2[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes2[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes2 = tf.transpose(box, (1, 2, 3, 4, 0))

        left_up = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        right_down = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        intersection = tf.maximum(right_down - left_up, 0.0)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        union_square = boxes1_square + boxes2_square - inter_square

        return tf.clip_by_value(1.0 * inter_square / union_square, 0.0, 1.0)

    def get_input(self):
        return self.net.images
