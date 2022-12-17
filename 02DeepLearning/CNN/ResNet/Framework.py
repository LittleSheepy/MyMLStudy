#  This is a framework which is used to implement the multiple GPU's when training a model.

import argparse
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import os


def get_gpus():
    value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    value = value.split(',')
    return len(value)


def make_dirs(path:str):
    pos = path.rfind("/")
    if pos < 0:
        raise Exception('Can not find the directory from the path', path)
    path = path[:pos]
    os.makedirs(path, exist_ok=True)


class Config:
    def __init__(self):
        self.lr = 0.01
        self.epoches = 2000
        self.batch_size = 10
        self.save_path = 'models/{name}/{name}'.format(name=self.get_name())
        self.sample_path = None
        self.logdir = 'logs/{name}'.format(name=self.get_name())
        self.new_model = False
        self.gpus = get_gpus()

    def get_name(self):
        raise Exception('get_name() is not re-defined.')

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s=%s' % (key, attrs[key]) for key in attrs]
        return ', '.join(result)

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (int, float, bool, str):
                result[attr] = value
        return result

    def from_cmd(self):
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            t = type(value)
            if t == bool:
                parser.add_argument('--' + attr, default=value, help='Default to %s' % value,
                                    action='store_%s' % ('false' if value else 'true'))
            else:
                parser.add_argument('--' + attr, type=t, default=value, help='Default to %s' % value)
        parser.add_argument('--call', type=str, default="train", help='Call method, default by train()')
        a = parser.parse_args()
        for attr in attrs:
            setattr(self, attr, getattr(a, attr))
        self.call(a.call)
    def call(self, name):
        if name == "train":
            self.train()
        else:
            self.test()
    def get_da_train(self):
        raise Exception("get_da_train not defined")
    def train(self):
        app = self.get_app()
        with app:
            app.train(self.get_da_train(), self.get_da_train())
    def get_da_test(self):
        raise Exception("get_da_test not defined")
    def test(self):
        print("====test========")
        # app = self.get_app()
        # with app:
        #     app.train(self.get_da_test())

    def get_tensors(self):
        return Tensors(self)

    def get_sub_tensors(self, gpu_index):
        """
        Get the sub tensors for the specified gpu.
        :param gpu_index: the index (based on zero) of the GPU
        :return: the sub tensors which has the property 'inputs'
        """
        raise Exception('The get_sub_tensors() is not defined.')

    def get_app(self):
        return App(self)

    def get_optimizer(self, lr):
        return tf.train.AdamOptimizer(lr)


class Tensors:
    """
    提供train_ops, summary, lr, sub_ts[i]: {inputs, losses, private_tensors}
    """
    def __init__(self, config: Config):
        self.config = config
        self.sub_ts = []
        # with tf.variable_scope(config.get_name(), reuse=tf.AUTO_REUSE):   # None, True, the 2nd method to reuse variables
        with tf.variable_scope(config.get_name()) as scope:
            for i in range(config.gpus):
                with tf.device('/gpu:%d' % i):
                    self.sub_ts.append(config.get_sub_tensors(i))
                    scope.reuse_variables()    # The 1st method to reuse variables
                    # tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('%s_train' % config.get_name()):
            with tf.device('/gpu:0'):
                losses = [ts.losses for ts in self.sub_ts]    #  [gpus, losses]
                self.losses = tf.reduce_mean(losses, axis=0)  # [losses]
                for i in range(len(losses[0])):
                    tf.summary.scalar('loss_%d' % i, self.get_loss_for_summary(self.losses[i]))
                self.summary = tf.summary.merge_all()

                self.lr = tf.placeholder(tf.float32, name='lr')
                opt = config.get_optimizer(self.lr)

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    grads = self.compute_grads(opt)
                    self.apply_grads(grads, opt)

    def get_loss_for_summary(self, loss):
        return loss

    def apply_grads(self, grads, opt):
        self.train_ops = [opt.apply_gradients(gs) for gs in grads]

    def compute_grads(self, opt):
        grads = [[opt.compute_gradients(loss) for loss in ts.losses] for ts in self.sub_ts]   # [gpus, losses]
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]

    def get_grads_mean(self, grads, loss_idx):
        # grads: [gpus, losses]
        grads = [gs[loss_idx] for gs in grads]  # [gpus]
        vars = [pair[1] for pair in grads[0]]
        result = []
        for i, var in enumerate(vars):
            result.append((tf.reduce_mean([gs[i][0] for gs in grads], axis=0), var))
        return result


class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            self.ts = config.get_tensors()
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(config=cfg, graph=graph)
            self.saver = tf.train.Saver()
            if config.new_model:
                self.session.run(tf.global_variables_initializer())
                print('Use a new empty model')
            else:
                try:
                    make_dirs(config.save_path)
                    self.saver.restore(self.session, config.save_path)
                    print("==="*50,"\n",'Restore model from %s successfully!' % config.save_path)
                except:
                    print("==="*50,"\n",'Fail to restore model from %s, use a new empty model instead!!!!!!' % config.save_path)
                    self.session.run(tf.global_variables_initializer())

    def close(self):
        self.session.close()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def train(self, ds_train, ds_validation=None):
        self.before_train()
        cfg = self.config
        ts  = self.ts
        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)
        batches = ds_train.num_examples // (cfg.batch_size * cfg.gpus)

        for epoch in range(cfg.epoches):
            self.before_epoch(epoch)
            for batch in range(batches):
                self.before_batch(epoch, batch)
                feed_dict = self.get_feed_dict(ds_train)
                if len(ts.train_ops) == 1:
                    _, summary = self.session.run([ts.train_ops[0], ts.summary], feed_dict)
                else:
                    for train_op in ts.train_ops:
                        self.session.run([train_op, feed_dict])
                    summary = self.session.run(ts.summary, feed_dict)
                writer.add_summary(summary, epoch * batches + batch)
                self.after_batch(epoch, batch)
            print('Epoch:', epoch, flush=True)
            self.after_epoch(epoch)
        self.after_train()

    def before_train(self):
        print('Training is started!', flush=True)

    def before_epoch(self, epoch):
        pass

    def before_batch(self, epoch, batch):
        pass

    def after_train(self):
        print('Training is finished!', flush=True)

    def after_epoch(self, epoch):
        self.save()

    def after_batch(self, epoch, batch):
        pass

    def save(self):
        make_dirs(self.config.save_path)
        self.saver.save(self.session, self.config.save_path)
        print('Save model into', self.config.save_path, flush=True)

    def test(self, ds_test):
        pass

    def get_feed_dict(self, ds):
        result = {self.ts.lr: self.config.lr}
        for i in range(self.config.gpus):
            values = ds.next_batch(self.config.batch_size)
            for tensor, value in zip(self.ts.sub_ts[i].inputs, values):
                result[tensor] = value
        return result


if __name__ == '__main__':
    cfg = Config()
    cfg.from_cmd()
    print('-' * 100)
    print(cfg)

    dss = read_data_sets(cfg.sample_path)

    app = cfg.get_app()
    with app:
        app.train(dss.train, dss.validation)
        app.test(dss.test)
