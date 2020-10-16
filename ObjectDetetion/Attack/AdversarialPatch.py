import numpy as np
import tensorflow as tf

from lib.tools.utils import *
from yolo.YOLOv3 import YOLOv3
from lib.datasets.BufferDS import BufferDS
from lib.datasets.imgsDS import imgsDS
from lib.datasets.ds_func import ds_func_read_img
from lib.config.Config import config as config

class AdversarialPatch():
    def __init__(self, model:YOLOv3):
        self.model = model
        with tf.variable_scope("AdversarialPatch"):
            self.patch = tf.get_variable("patch", [30,30,3], tf.float32, trainable=True, initializer=tf.zeros_initializer())
        cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = tf.Session(config=cfg)

    def train(self):
        self.input_data   = tf.placeholder(dtype=tf.float32,shape=[None, 416, 416, 3], name='input_data')
        inputs_overlay = self.overlay_patch_random(self.patch, self.input_data)

        start_vars = set(x.name for x in tf.global_variables())
        self.preds = self.model.build_network(inputs_overlay)
        end_vars = tf.global_variables()
        self.net_var = [x for x in end_vars if x.name not in start_vars]
        #conv_lbbox, conv_mbbox, conv_sbbox = self.model.build_network(img_data)
        self.loss = self.compute_loss()

        opt = tf.train.AdamOptimizer()
        gradients = opt.compute_gradients(self.loss, [self.patch])
        # gradients = [(-g[0], g[1]) for g in gradients]
        train_op = opt.apply_gradients(gradients)
        self.sess.run(tf.global_variables_initializer())
        self.loader = tf.train.Saver(self.net_var)
        try:
            print('=> Restoring weights from: %s ... ' % config.original_model)
            self.loader.restore(self.sess, config.original_model)
        except Exception:
            print('=> %s does not exist !!!' % config.original_model)
            print('=> Now it starts to train YOLOV3 from scratch ...')

        self.data_train = imgsDS()
        self.BuffDS =BufferDS(self.data_train, ds_func_read_img)
        for i in range(2):
            img_data = self.BuffDS.get()
            _, patch_v, gradients_v = self.sess.run([train_op,self.patch,gradients], feed_dict={self.input_data: img_data})
            print(i,gradients_v)
    def overlay_patch_random(self, patch, inputs):
        inputs_shape = get_shape(inputs)
        patch_shape = get_shape(patch)
        lpad = tf.random_uniform([], 0, inputs_shape[2] - patch_shape[1])
        rpad = inputs_shape[2] - patch_shape[1] - lpad
        upad = tf.random_uniform([], 0, inputs_shape[1] - patch_shape[0])
        dpad = inputs_shape[1] - patch_shape[0] - upad
        patch_paded = tf.pad(patch, [[lpad, rpad],[upad, dpad],[0,0]])
        patch_paded = tf.expand_dims(patch_paded, axis=0)

        patch_mask_shape = (1, inputs_shape[1], inputs_shape[2], inputs_shape[3])       # (1,416,416,3)
        patch_mask = np.zeros(patch_mask_shape)
        patch_mask[:,100:130, 100:130,:] = 1
        patch_mask = tf.cast(patch_mask, inputs.dtype)

        img_mask = 1 - patch_mask

        inputs_overlay = inputs * img_mask + patch_paded * patch_mask

        return inputs_overlay
    def compute_loss(self):
        loss = 0
        conv_shape = get_shape(self.preds[0])
        batch_size = tf.cast(conv_shape[0], tf.float32)
        for i, pred in enumerate(self.preds):
            pred = tf.reshape(pred, (-1 , 5 + config.num_class))  # (batch_size, -1, 85)
            conv_raw_conf = pred[:, 4]  # 置信度
            lable_shape = get_shape(conv_raw_conf)
            lable = tf.zeros(lable_shape, tf.float32)
            conf_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=lable, logits=conv_raw_conf)
            conf_loss = tf.reduce_sum(conf_loss, axis=-1)/batch_size
            loss += conf_loss
        return loss






if __name__ == '__main__':
    yolov3 = YOLOv3()
    ap = AdversarialPatch(yolov3)
    ap.train()