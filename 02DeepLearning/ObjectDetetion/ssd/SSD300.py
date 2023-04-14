import cv2
import time
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tf_slim as slim

from lib.net.DarkNet53 import DarkNet53
from lib.config.Config import config as config
from lib.datasets.pascalVocDS import pascalVocDS as pascalVocDS
from lib.tools.utils import *
from lib.tools.bbox import *
from lib.tools.custom_layers import *
import lib.tools.utils
from lib.datasets.dataset import Dataset
from lib.datasets.BufferDS import BufferDS
from lib.datasets.ds_func import ds_func_SSD


class SSD300():
    def __init__(self):
        self.end_points = {}
        self.logits = []
        self.logits_softmax = []
        self.localisations = []
        self.gclasses = []
        self.glocalisations = []
        self.gscores = []
        cfg1 = tf.ConfigProto(allow_soft_placement=True)
        cfg1.gpu_options.allow_growth = True
        self.sess                = tf.Session(config=cfg1)

    def ssd_arg_scope(self,weight_decay=0.0005, data_format='NHWC'):
        """Defines the VGG arg scope.
        Args:
          weight_decay: The l2 regularization coefficient.
        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                data_format=data_format) as sc:
                return sc
    def build_network(self):
        end_points = self.end_points
        dropout_keep_prob = config.dropout_keep_prob
        is_training = config.training
        # Original VGG-16 blocks.
        net = slim.repeat(self.inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')           # (?, 300, 300, 64)
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')                                   # (?, 150, 150, 64)
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')                  # (?, 150, 150, 128)
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')                                   # (?, 75, 75, 128)
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')                  # (?, 75, 75, 256)
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')                                   # (?, 38, 38, 256)
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')                  # (?, 38, 38, 512)
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')                                   # (?, 19, 19, 512)
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')                  # (?, 19, 19, 512)
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!让我们把它放大!
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')                         # (?, 19, 19, 1024)
        end_points['block6'] = net
        net = tf.layers.dropout(net, rate=config.dropout_keep_prob, training=config.training)
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        # Prediction and localisations layers.
        normalizations = [20, -1, -1, -1, -1, -1]
        for i, layer in enumerate(config.feature_layers):
            with tf.variable_scope(layer + '_box'):
                net = end_points[layer]
                net_shape = get_shape(net, 4)
                if normalizations[i] > 0:
                    with tf.variable_scope("L2Normalization"):
                        net = tf.nn.l2_normalize(net, 3)
                        scale = slim.model_variable("gamma", net_shape[-1], net.dtype.base_dtype, tf.ones_initializer())
                        net = tf.multiply(net, scale)
                # Location.
                num_loc_pred = config.num_anchors[i] * 4
                loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, scope='conv_loc')
                loc_pred = tf.reshape(loc_pred, [-1, net_shape[1], net_shape[2], config.num_anchors[i], 4])     # (?, 38, 38, 4, 4)
                # Class prediction.
                num_cls_pred = config.num_anchors[i] * config.num_class
                cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, scope='conv_cls')
                cls_pred = tf.reshape(cls_pred, [-1, net_shape[1], net_shape[2], config.num_anchors[i], config.num_class])  # (?, 38, 38, 4, 21)
            self.logits.append(cls_pred)
            self.logits_softmax.append(tf.nn.softmax(cls_pred))
            self.localisations.append(loc_pred)
    def define_lable(self):
        for i in range(6):
            shape = config.feature_shapes[i]
            num_anchor = config.num_anchors[i]
            self.gclasses.append(tf.placeholder(dtype=tf.int64, shape=[None, shape, shape, num_anchor], name='gclass%d'%shape))
            self.glocalisations.append(tf.placeholder(dtype=tf.float32, shape=[None, shape, shape, num_anchor, 4], name='gloc%d'%shape))
            self.gscores.append(tf.placeholder(dtype=tf.float32, shape=[None, shape, shape, num_anchor], name='gscore%d'%shape))

    def compute_loss(self):
        """
        logits ： [[1,38,38,4,21],[],[],[],[],[]] 466444
        :return:
        """
        match_threshold = 0.5
        negative_ratio = 3.
        alpha = 1.
        logits = self.logits
        localisations = self.localisations
        gclasses = self.gclasses
        glocalisations = self.glocalisations
        gscores = self.gscores

        with tf.name_scope('ssd_losses', 'ssd_losses'):
            lshape = get_shape(logits[0], 5)
            num_classes = lshape[-1]
            batch_size = lshape[0]

            # Flatten out all vectors!
            flogits = []
            fgclasses = []
            fgscores = []
            flocalisations = []
            fglocalisations = []
            for i in range(len(logits)):
                flogits.append(tf.reshape(logits[i], [-1, num_classes]))
                fgclasses.append(tf.reshape(gclasses[i], [-1]))
                fgscores.append(tf.reshape(gscores[i], [-1]))
                flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
                fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
            # And concat the crap!
            logits = tf.concat(flogits, axis=0)
            gclasses = tf.concat(fgclasses, axis=0)
            gscores = tf.concat(fgscores, axis=0)
            localisations = tf.concat(flocalisations, axis=0)
            glocalisations = tf.concat(fglocalisations, axis=0)
            dtype = logits.dtype

            # Compute positive matching mask...
            pmask = gscores > match_threshold
            fpmask = tf.cast(pmask, dtype)
            n_positives = tf.reduce_sum(fpmask)

            # Hard negative mining...
            no_classes = tf.cast(pmask, tf.int32)
            predictions = slim.softmax(logits)
            nmask = tf.logical_and(tf.logical_not(pmask), gscores > -0.5)
            fnmask = tf.cast(nmask, dtype)
            nvalues = tf.where(nmask, predictions[:, 0], 1. - fnmask)
            nvalues_flat = tf.reshape(nvalues, [-1])
            # Number of negative entries to select.
            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
            n_neg = tf.minimum(n_neg, max_neg_entries)

            val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
            max_hard_pred = -val[-1]
            # Final negative mask.
            nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
            fnmask = tf.cast(nmask, dtype)
            batch_size = tf.cast(batch_size, dtype)
            # Add cross-entropy loss.
            with tf.name_scope('cross_entropy_pos'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)
                loss = tf.reduce_sum(loss * fpmask)
                loss_pos = tf.div(loss, batch_size, name='value')

            with tf.name_scope('cross_entropy_neg'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=no_classes)
                loss_neg = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')

            # Add localization loss: smooth L1, L2, ...
            with tf.name_scope('localization'):
                # Weights Tensor: positive mask + random negative.
                weights = tf.expand_dims(alpha * fpmask, axis=-1)
                loss = abs_smooth(localisations - glocalisations)
                loss_loc = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
                #tf.losses.add_loss(loss)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss =  tf.reduce_sum(regularization_losses)
            self.loss = tf.reduce_sum([loss_pos, loss_neg, loss_loc, regularization_loss])
    def train(self):
        #self.global_step = tf.Variable(1, dtype=tf.int64, trainable=False, name='global_step')
        self.inputs   = tf.placeholder(dtype=tf.float32,shape=[None, 300, 300, 3], name='input_data')
        with tf.variable_scope("ssd_300_vgg", 'ssd_300_vgg'):
            with slim.arg_scope(self.ssd_arg_scope()):
                self.build_network()
        self.net_var = tf.global_variables()
        self.define_lable()
        self.compute_loss()
        self.learn_rate = config.lr
        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        self.train_op = optimizer.minimize(self.loss)
        self.loader = tf.train.Saver(self.net_var)
        saver_variables = tf.global_variables()
        self.saver  = tf.train.Saver(saver_variables)
        tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(config.logdir, graph=self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.data_train = pascalVocDS()
        self.BuffDS =BufferDS(self.data_train, ds_func_SSD)
        try:
            print('=> Restoring weights from: %s ... ' % config.original_model)
            self.loader.restore(self.sess, config.original_model)
        except Exception:
            print('=> %s does not exist !!!' % config.original_model)
            print('=> Now it starts to train YOLOV3 from scratch ...')
        for iter in range(config.max_iter):
            print("iter", iter)
            inputs, locs_data, scores_data, classes_data = self.BuffDS.get()
            feed_dict = {self.inputs:np.array(inputs).reshape(-1, 300, 300, 3)}
            for i in range(6):
                shape = config.feature_shapes[i]
                num_anchor = config.num_anchors[i]
                feed_dict.update({self.gclasses[i]:np.array(classes_data[i]).reshape([-1, shape, shape, num_anchor]),
                                  self.glocalisations[i]: np.array(locs_data[i]).reshape([-1, shape, shape, num_anchor, 4]),
                                  self.gscores[i]:np.array(scores_data[i]).reshape([-1, shape, shape, num_anchor])
                                  })
            _, summary_op = self.sess.run([self.train_op, self.summary_op], feed_dict=feed_dict)
    def test_image(self, img_file=None):
        config.training = False
        self.inputs_org   = tf.placeholder(dtype=tf.float32,shape=[None, None, None, 3], name='input_data')
        self.inputs = tf.image.resize_images(self.inputs_org, (300, 300), tf.image.ResizeMethod.BILINEAR)
        with tf.variable_scope("ssd_300_vgg", 'ssd_300_vgg'):
            with slim.arg_scope(self.ssd_arg_scope()):
                self.build_network()
        if not img_file:
            img_file = config.data_test_img
        org_img = cv2.imread(img_file)
        org_h, org_w, _ = org_img.shape
        org_img1 = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = preprocess_image(org_img, (config.image_size, config.image_size), "resize")
        self.net_var = tf.global_variables()
        self.sess.run(tf.global_variables_initializer())
        self.loader = tf.train.Saver(self.net_var)
        self.loader.restore(self.sess, config.original_model)
        pred_logits, pred_logits_softmax, pred_locs = self.sess.run(
            [self.logits, self.logits_softmax, self.localisations], feed_dict={self.inputs_org: [org_img1]}
        )

        anchors = generate_anchors_by_size_ratios()
        grids = generate_grids(config.strides)
        bboxes, scores, classes = [], [], []
        for i , [pred_locs_layer, pred_logits_layer, anchors_layer] in enumerate(zip(pred_locs, pred_logits_softmax, anchors)):
            pred_locs_layer, pred_logits_layer = pred_locs_layer[0], pred_logits_layer[0]
            feature_shape = config.feature_shapes[i]
            n_bbox = feature_shape * feature_shape * config.num_anchors[i]
            n_classes = config.num_class
            # decode bbox
            bboxes_l = bboxes_decode(pred_locs_layer, anchors_layer, grids[i], [feature_shape, feature_shape])    # 1 38 38 4 4
            bboxes_l = bboxes_l.reshape([n_bbox, 4])        # (1, 5776, 4)
            # score & classes
            pred_logits_layer = pred_logits_layer.reshape([n_bbox, n_classes])
            pred_logits_layer = pred_logits_layer[:, 1:]                 # (5776, 20)
            scores_l = np.max(pred_logits_layer,axis=-1)                             # (5776)
            classes_l = np.argmax(pred_logits_layer, axis=-1)                 # (5776)
            bboxes.append(bboxes_l)
            scores.append(scores_l)
            classes.append(classes_l)

        bboxes = np.concatenate(bboxes)
        scores = np.concatenate(scores)
        classes = np.concatenate(classes)

        bboxes, scores, classes = scores_select(bboxes, scores, classes)
        bboxes *= [org_w, org_h, org_w, org_h]

        classes, scores, bboxes = bboxes_nms(classes, scores, bboxes, nms_threshold=0.5, method="soft-nms")
        scores = np.expand_dims(scores, axis=-1)
        classes = np.expand_dims(classes, axis=-1)
        bboxes = np.concatenate([bboxes, scores, classes], axis=-1)
        #bboxes = nms(bboxes)
        img_detection = draw_bbox(org_img, bboxes)
        # cv2.imwrite(self.write_image_path + image_name, image)
        cv2.imshow("detection_results", img_detection)  # 显示图片
        cv2.waitKey(0)  # 等待按任意键退出


        a = 1


    def decode(self):
        pass


if __name__ == '__main__':
    ssd = SSD300()
    ssd.train()
    #ssd.test_image()
    print(ssd.end_points)
