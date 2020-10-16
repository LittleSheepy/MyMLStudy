import tensorflow as tf
import numpy as np
import pickle
import random

print(tf.__version__)

class Config:
    def __init__(self):
        self.save_path = "models/AlexNet"
        self.logdir = "logs"
        self.sample_path = ""
        self.lr = 0.001
        self.epoches = 1000
        self.batch_size = 500
        self.classes = 17

class Tensors:
    def __init__(self, classes):
        with tf.device("/gpu:0"):
            self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="x")

            layer = tf.layers.conv2d(self.x, filters=96, kernel_size=11, strides=4, padding="same", activation=tf.nn.relu)   # (?, 56, 56,96)
            layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2, padding="same")      # (?, 28, 28,96)
            layer = tf.nn.lrn(layer, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75)         # (?, 28, 28,96)

            layer = tf.layers.conv2d(layer, filters=256, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)   # (?, 28, 28,256)
            layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2, padding="same")      # (?, 14, 14,256)
            layer = tf.nn.lrn(layer, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75)         # (?, 14, 14,256)

            layer = tf.layers.conv2d(layer, filters=384, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)   # (?, 14, 14,384)

            layer = tf.layers.conv2d(layer, filters=384, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)   # (?, 14, 14,384)

            layer = tf.layers.conv2d(layer, filters=256, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)   # (?, 14, 14,256)
            layer = tf.layers.max_pooling2d(layer, pool_size=3, strides=2, padding="same")      # (?, 7, 7,256)
            layer = tf.nn.lrn(layer, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75)         # (?, 7, 7,256)

            layer = tf.reshape(layer, [-1, 7*7*256])
            layer = tf.layers.dense(layer, 4096, activation=tf.nn.tanh)
            layer = tf.layers.dropout(layer, 0.5)
            layer = tf.layers.dense(layer, 4096, activation=tf.nn.tanh)
            layer = tf.layers.dropout(layer, 0.5)
            self.vect = layer

            self.logits = tf.layers.dense(self.vect, classes, activation=tf.nn.softmax)
            self.y = tf.placeholder(tf.int32, [None], name="y")
            self.y_hot = tf.one_hot(self.y, classes)
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_hot, logits=self.logits)
            self.loss = tf.reduce_mean(self.loss)
            self.loss_summary = tf.summary.scalar("loss", self.loss)

            self.y_pre = tf.argmax(self.logits, axis=1, output_type=tf.int32)
            self.y = tf.argmax(self.y_hot, axis=1, output_type=tf.int32)

            self.precise = tf.cast(tf.equal(self.y, self.y_pre),tf.float32)
            self.precise = tf.reduce_mean(self.precise)
            self.precise_summary = tf.summary.scalar("precise", self.precise)

            self.lr = tf.placeholder(tf.float32, shape=[], name="lr")
            opt = tf.train.MomentumOptimizer(self.lr, 0.9)
            # opt = tf.train.AdamOptimizer(self.lr)
            # opt = tf.train.GradientDescentOptimizer(self.lr)

            self.train_op = opt.minimize(self.loss)

class DS:
    def __init__(self, xs, ys, shuffle=False):
        self.xs = xs
        self.ys = ys
        self.N = len(self.ys)
        self.pos = 0
        self.num = 0
        if shuffle:self.shuffle()
    def next_batch(self, batch_size):
        next = self.pos + batch_size
        if next < self.N:
            xs = self.xs[self.pos:next]
            ys = self.ys[self.pos:next]
            self.pos = next
        else:
            xs = self.xs[self.pos:]
            ys = self.ys[self.pos:]
            self.pos = 0
            self.num += 1
        if self.num % 3 == 0:
            self.shuffle()
        return xs, ys
    def shuffle(self):
        ds = [(x, y) for x, y in zip(self.xs, self.ys)]
        random.shuffle(ds)
        self.xs, self.ys = [], []
        for x, y in ds:
            self.xs.append(x)
            self.ys.append(y)



class AlexNet:
    def __init__(self, cfg:Config=None):
        self.cfg = cfg or Config()
        graph = tf.Graph()
        with graph.as_default():
            self.ts = Tensors(self.cfg.classes)
            cfgPro = tf.ConfigProto()
            cfgPro.allow_soft_placement = True
            cfgPro.gpu_options.allow_growth = True
            self.session = tf.Session(config=cfgPro, graph=graph)
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.session, self.cfg.save_path)
                print(f"Restore model from {self.cfg.save_path} successfully!")
            except:
                self.session.run(tf.global_variables_initializer())
                print(f"Fail to restore model from {self.cfg.save_path}, use a new empty model instead!!!!")
    def train(self, xs, ys):
        cfg = self.cfg
        ts = self.ts
        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)
        batches = len(xs) // cfg.batch_size + 1
        ds = DS(xs, ys, shuffle=True)
        for epoch in range(cfg.epoches):
            for batch in range(batches):
                x, y = ds.next_batch(cfg.batch_size)
                _, loss_summary, precise_summary= self.session.run([ts.train_op, ts.loss_summary, ts.precise_summary], {ts.x:x, ts.y_hot:y, ts.lr:cfg.lr})
                writer.add_summary(loss_summary, epoch * batches + batch)
                writer.add_summary(precise_summary, epoch * batches + batch)
            self.saver.save(self.session, cfg.save_path)
            print(f"saved, epoch={epoch}")


if __name__ == '__main__':
    xs, ys = pickle.load(open('dataset.pkl', 'rb'))
    print(len(ys))
    alexNet = AlexNet()
    alexNet.train(xs, ys)



