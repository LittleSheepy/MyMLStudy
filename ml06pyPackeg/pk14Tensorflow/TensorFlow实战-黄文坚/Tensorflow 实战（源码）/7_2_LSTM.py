#%%
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import time
import numpy as np
import tensorflow as tf
import reader

#flags = tf.flags
#logging = tf.logging



#flags.DEFINE_string("save_path", None,
#                    "Model output directory.")
#flags.DEFINE_bool("use_fp16", False,
#                  "Train using 16-bit floats instead of 32bit floats")

#FLAGS = flags.FLAGS


#def data_type():
#  return tf.float16 if FLAGS.use_fp16 else tf.float32

# 定义语言模型处理输入数据
class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps       # LSTM的展开步数
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


# 定义语言模型的class
class PTBModel(object):
  """The PTB model."""
 # is_training：训练标记
  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size     # LSTM 的节点数
    vocab_size = config.vocab_size    # 词汇表的大小

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    # （翻译）forget gate biases初始化为1可以获得稍微好一些的结果，但这样会和论文结果不同
    # 首先使用BasicLSTMCell定义单个基本的LSTM单元。这里的size其实就是hidden_size。
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    # 多层LSTM结构 这里使用了两层config.num_layers = 2
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    # 输入预处理
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=tf.float32)
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, 0.5)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)# output: shape[num_steps][batch,hidden_size]

    # 把之前的list展开，成[batch, hidden_size*num_steps],然后 reshape, 成[batch*numsteps, hidden_size]
    # [num_steps][batch,hidden_size] ==>[batch, hidden_size*num_steps] ==> [batch*numsteps, hidden_size]
    output = tf.reshape(tf.concat(outputs, 1), [-1, size])

    #=====损失函数计算=======
    # softmax_w , shape=[hidden_size, vocab_size], 用于将distributed表示的单词转化为one-hot表示
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
    # [batch*numsteps, vocab_size] 从隐藏语义转化成完全表示
    logits = tf.matmul(output, softmax_w) + softmax_b
    # loss , shape=[batch*num_steps]
    # 带权重的交叉熵计算
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],# output [batch*numsteps, vocab_size]
        [tf.reshape(input_.targets, [-1])], # target, [batch_size, num_steps] 然后展开成一维【列表】
        [tf.ones([batch_size * num_steps], dtype=tf.float32)])
    self._cost = cost = tf.reduce_sum(loss) / batch_size   # 计算得到平均每批batch的误差
    self._final_state = state

    if not is_training:
      return

    # 生成一个lr的variable，但是trainable=False，也就是不进行求导。
    self._lr = tf.Variable(0.0, trainable=False)
    # gradients: return A list of sum(dy/dx) for each x in xs.
    tvars = tf.trainable_variables()
    # tf.gradients 用来计算导数
    # clip_by_global_norm  修正梯度值，用于控制梯度爆炸的问题。
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)

    optimizer = tf.train.GradientDescentOptimizer(self._lr)

    self._train_op = optimizer.apply_gradients(zip(grads, tvars),
                                      global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1          # 网络中权重初始scale
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200         # 隐藏层中单元数目
  max_epoch = 4
  max_max_epoch = 13        # 指的是整个文本循环次数。
  keep_prob = 0.5
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000        # 词典规模，总共10K个词


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)




raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph().as_default():
  initializer = tf.random_uniform_initializer(-config.init_scale,
                                              config.init_scale)

  with tf.name_scope("Train"):
    train_input = PTBInput(config=config, data=train_data, name="TrainInput")
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config, input_=train_input)
      #tf.scalar_summary("Training Loss", m.cost)
      #tf.scalar_summary("Learning Rate", m.lr)

  with tf.name_scope("Valid"):
    valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      #tf.scalar_summary("Validation Loss", mvalid.cost)

  with tf.name_scope("Test"):
    test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
      mtest = PTBModel(is_training=False, config=eval_config,
                       input_=test_input)

  sv = tf.train.Supervisor()
  with sv.managed_session() as session:
    for i in range(config.max_max_epoch):
      # 先计算学习速率衰减值
      # 在 遍数小于max epoch时， lr_decay = 1 ; > max_epoch时， lr_decay = 0.5^(i-max_epoch)
      lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay) # 设置learning rate

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid)
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    test_perplexity = run_epoch(session, mtest)
    print("Test Perplexity: %.3f" % test_perplexity)

     # if FLAGS.save_path:
     #   print("Saving model to %s." % FLAGS.save_path)
     #   sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

#if __name__ == "__main__":
#  tf.app.run()