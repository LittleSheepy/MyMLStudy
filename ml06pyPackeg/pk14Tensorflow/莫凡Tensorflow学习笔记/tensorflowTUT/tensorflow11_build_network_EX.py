# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

prediction_value = []
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        prediction_value.append(sess.run(prediction, feed_dict={xs: x_data}))

# 生成子图，相当于fig = plt.figure(),ax = fig.add_subplot(),
# 其中ax的函数参数表示把当前画布进行分割，
# 例：fig.add_subplot(2,2,2).表示将画布分割为两行两列
fig, ax = plt.subplots()
#ax在第2个子图中绘制，其中行优先，
xdata, ydata = [], []      #初始化两个数组

#第三个参数表示画曲线的颜色和线型，具体参见：https://blog.csdn.net/tengqingyong/article/details/78829596
ln, = ax.plot([], [], 'r-', animated=False)
def init():
    ax.scatter(x_data, y_data)
    return ln,               #返回曲线

def update(n):
    ln.set_data(x_data, prediction_value[n])
    return ln,

#这里的frames在调用update函数是会将frames作为实参传递给“n”
ani = FuncAnimation(fig, update, frames=range(0, 20),
                    init_func=init, blit=False)
ani.save('line.gif', dpi=80, writer='imagemagick')
#ani.save('GifDemo.htm', dpi=80, writer='imagemagick')
plt.show()