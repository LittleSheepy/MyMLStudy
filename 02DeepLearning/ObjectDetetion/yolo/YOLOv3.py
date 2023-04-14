import cv2
import time
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from lib.net.DarkNet53 import DarkNet53
from lib.config.Config import config as config
from lib.datasets.pascal_voc import pascal_voc
from lib.tools.utils import *
from lib.datasets.dataset import Dataset

class YOLOv3():
    def init(self):
        cfg1 = tf.ConfigProto(allow_soft_placement=True)
        cfg1.gpu_options.allow_growth = True
        self.sess                = tf.Session(config=cfg1)
        self.input_data   = tf.placeholder(dtype=tf.float32,shape=[None, 416, 416, 3], name='input_data')
        self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.build_network(self.input_data)

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, config.anchors[0], config.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, config.anchors[1], config.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, config.anchors[2], config.strides[2])

    def build_network(self, input_data):
        self.dk53 = DarkNet53(input_data)
        # large 13x13
        net = conv2d(self.dk53.output, 512, 1, 0, name="conv53")
        net = conv2d(net, 1024, 3, 1, name="conv54")
        net = conv2d(net, 512, 1, 0, name="conv55")
        net = conv2d(net, 1024, 3, 1, name="conv56")
        net = conv2d(net, 512, 1, 0, name="conv57")     # (?, 13, 13, 512)
        conv_lobj_branch = conv2d(net, 1024, 3, 1, name="conv_lobj_branch")
        conv_lbbox = conv2d(conv_lobj_branch, 3*(config.num_class + 5), 1, 0, name="conv_lbbox",
                            batch_normalize=False, activation=None, use_bias=True)
        # middle 26x26
        net = conv2d(net, 256, 1, 0, name="conv58")
        net = upsample(net, name='upsample0', method=config.upsample_method)
        net = route("route_middle", net, self.dk53.route_2)     # (?, 26, 26, 768=256+512)
        net = conv2d(net, 256, 1, 0)
        net = conv2d(net, 512, 3, 1)
        net = conv2d(net, 256, 1)
        net = conv2d(net, 512, 3, 1)
        net = conv2d(net, 256, 1, 0)     # (?, 26, 26, 256)
        conv_mobj_branch = conv2d(net, 512, 3, 1, name="conv_mobj_branch")
        conv_mbbox = conv2d(conv_mobj_branch, 3*(config.num_class + 5), 1, 0, name="conv_mbbox",
                            batch_normalize=False, activation=None, use_bias=True)
        # small 52x52
        net = conv2d(net, 128, 1, 0, name="conv64")
        net = upsample(net, name='upsample0', method=config.upsample_method)
        net = route("route_middle", net, self.dk53.route_1)     # (?, 52, 52, 384=128+256)
        net = conv2d(net, 128, 1, 0)
        net = conv2d(net, 256, 3, 1)
        net = conv2d(net, 128, 1)
        net = conv2d(net, 256, 3, 1)
        net = conv2d(net, 128, 1, 0)     # (?, 52, 52, 128)
        conv_sobj_branch = conv2d(net, 256, 3, 1, name="conv_sobj_branch")
        conv_sbbox = conv2d(conv_sobj_branch, 3*(config.num_class + 5), 1, 0, name="conv_sbbox",
                            batch_normalize=False, activation=None, use_bias=True)
        return conv_lbbox, conv_mbbox, conv_sbbox

    def xy_grid(self, output_size):
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)        # (?, ?, 2) (52, 52, 2)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [config.batch_size, 1, 1, config.anchor_per_scale, 1])    # (?, ?, ?, 3, 2) (?, 52, 52, 3, 2)
        return tf.cast(xy_grid, tf.float32)


    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        # conv_shape       = tf.shape(conv_output)
        conv_shape = get_shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + config.num_class))     # (?, 52, 52, 3, 85)

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]    # 坐标XY (?, ?, ?, 3, 2) (?, 52, 52, 3, 2)
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]    # 框WH
        conv_raw_conf = conv_output[:, :, :, :, 4:5]    # 置信度
        conv_raw_prob = conv_output[:, :, :, :, 5: ]    # 类别

        xy_grid = self.xy_grid(output_size)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride    # (?, ?, ?, 3, 2) (?, 52, 52, 3, 2)
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def define_lable(self):
        with tf.name_scope('define_lable'):
            self.label_sbbox  = tf.placeholder(dtype=tf.float32,shape=[None, None, None, 3, 85], name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32,shape=[None, None, None, 3, 85], name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32,shape=[None, None, None, 3, 85], name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32,shape=[None, 150, 4], name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32,shape=[None, 150, 4], name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32,shape=[None, 150, 4], name='lbboxes')
    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou
    # self.conv_sbbox, label_sbbox, true_sbbox,
    def loss_layer(self, conv, label, bboxes, anchors, stride):

        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size, config.anchor_per_scale, 5 + config.num_class))    # (?, ?, ?, 3, 85)
        conv_raw_conf = conv[:, :, :, :, 4:5]   # 置信度
        conv_raw_prob = conv[:, :, :, :, 5:]    # 类别

        pred = self.decode(conv, anchors, stride)
        pred_xywh     = pred[:, :, :, :, 0:4]   # sigmoid后的XYWH

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]  #　置信度
        label_prob    = label[:, :, :, :, 5:]   #　类别

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < config.iou_loss_thresh, tf.float32 )

        conf_focal = self.focal(respond_bbox, tf.sigmoid(conv_raw_conf))

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss
    def compute_loss(self):
        loss_sbbox = self.loss_layer(self.conv_sbbox, self.label_sbbox, self.true_sbboxes, config.anchors[0], config.strides[0])
        loss_mbbox = self.loss_layer(self.conv_mbbox, self.label_mbbox, self.true_mbboxes, config.anchors[0], config.strides[1])
        loss_lbbox = self.loss_layer(self.conv_lbbox, self.label_lbbox, self.true_lbboxes, config.anchors[0], config.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss

    def learn_rate(self):
        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(config.warmup_epochs * self.steps_per_epochs,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((config.first_stage_epochs + config.second_stage_epochs)* self.steps_per_epochs,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * config.lr_init,
                false_fn=lambda: config.lr_end + 0.5 * (config.lr_init - config.lr_end) *
                                (1 + tf.cos((self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            self.global_step_update = tf.assign_add(self.global_step, 1.0)
    def define_train_op(self):
        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(config.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, self.global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, self.global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()
    def define_saver(self):
        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate",      self.learn_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = config.logs
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)
    def train(self):
        self.init()
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_epochs    = len(self.trainset)


        self.net_var = tf.global_variables()
        self.define_lable()
        self.giou_loss, self.conf_loss, self.prob_loss = self.compute_loss()
        self.loss = self.giou_loss + self.conf_loss + self.prob_loss
        self.learn_rate()
        self.define_train_op()
        self.define_saver()
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % config.original_model)
            self.loader.restore(self.sess, config.original_model)
        except Exception:
            print('=> %s does not exist !!!' % config.original_model)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            config.first_stage_epochs = 0

        for epoch in range(1, 1+config.first_stage_epochs+config.second_stage_epochs):
            if epoch <= config.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_losses, test_epoch_losses = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                self.input_data:   train_data[0],   # (1, 416, 416, 3)
                                                self.label_sbbox:  train_data[1],   # (1, 52, 52, 3, 85)
                                                self.label_mbbox:  train_data[2],   # (1, 26, 26, 3, 85)
                                                self.label_lbbox:  train_data[3],   # (1, 13, 13, 3, 85)
                                                self.true_sbboxes: train_data[4],   # (1, 150, 4)
                                                self.true_mbboxes: train_data[5],   # (1, 150, 4)
                                                self.true_lbboxes: train_data[6],   # (1, 150, 4)
                })

                train_epoch_losses.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" %train_step_loss)

            for test_data in self.testset:
                test_step_loss = self.sess.run( self.loss, feed_dict={
                                                self.input_data:   test_data[0],
                                                self.label_sbbox:  test_data[1],
                                                self.label_mbbox:  test_data[2],
                                                self.label_lbbox:  test_data[3],
                                                self.true_sbboxes: test_data[4],
                                                self.true_mbboxes: test_data[5],
                                                self.true_lbboxes: test_data[6],
                })

                test_epoch_losses.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_losses), np.mean(test_epoch_losses)
            ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)
    def test_image(self, img_file=None):
        import pprint
        self.init()
        config.training = False
        if not img_file:
            img_file = config.data_test_img
        org_img = cv2.imread(img_file)
        org_h, org_w, _ = org_img.shape
        image = preprocess_image(org_img, (config.test_image_size, config.test_image_size))
        self.net_var = tf.global_variables()
        self.sess.run(tf.global_variables_initializer())
        self.loader = tf.train.Saver(self.net_var)
        self.loader.restore(self.sess, config.original_model)
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox], feed_dict={self.input_data: image}
        )
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + config.num_class)),
                                    np.reshape(pred_mbbox, (-1, 5 + config.num_class)),
                                    np.reshape(pred_lbbox, (-1, 5 + config.num_class))], axis=0)
        bboxes = postprocess_boxes(pred_bbox, (org_h, org_w), config.test_image_size)
        bboxes = nms(bboxes, method='soft-nms')
        img_detection = draw_bbox(org_img, bboxes)
        pprint.pprint(bboxes)
        # cv2.imwrite(self.write_image_path + image_name, image)
        cv2.imshow("detection_results", img_detection)  # 显示图片
        cv2.waitKey(0)  # 等待按任意键退出









if __name__ == '__main__':
    yolov3 = YOLOv3()
    #yolov3.train()
    yolov3.test_image()
    a=1