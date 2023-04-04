# encoding:utf-8
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 下载并导入MNIST数据集
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# MNIST数据集每张图片的大小是28×28，
# 这里将输入x分成28个时间段，每个时间段的内容为28个值
n_input = 28
n_steps = 28
# 隐含层
n_hidden = 128
# 0~9共10个分类
n_classes = 10

# 定义占位符
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# 将原始的28×28数据调成具有28个时间段的list，每个list是一个1×28数组，
# 将这28个时序送入RNN中
x1 = tf.unstack(x, n_steps, 1)

# cell类，这里使用LSTM，BasicLSTMCell是LSTM的basic版本
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# 通过cell类构建RNN，这里使用静态RNN
outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)

# 全连接层
pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)

# 定义学习率，训练次数，batch长度
learning_rate = 0.001
training_iters = 20000
batch_size = 10

# 定义损失和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 测试准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 启动session
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    step = 1
    # 开始训练
    while step * batch_size < training_iters:
        # 批量获取MNIST数据
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % 100 == 0:
            # 计算批次数据的准确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print(" Finished!")

    # 计算准确率
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_data, y: test_label}))