import tensorflow as tf

w1 = tf.Variable(2.0)
w2 = tf.Variable(88.)
w3 = tf.Variable(66.)

a = tf.multiply(w1, w2)
a_stop = tf.stop_gradient(a)

b = tf.multiply(w2, 3.)
c = tf.add(w3, a)  # 把a换成换成a_stop之后，w1和a的梯度都为None
# [None, <tf.Tensor 'gradients/Mul_1_grad/Reshape:0' shape=() dtype=float32>, <tf.Tensor 'gradients/Add_grad/Reshape:0' shape=() dtype=float32>, None, <tf.Tensor 'gradients/Add_1_grad/Reshape_1:0' shape=() dtype=float32>, <tf.Tensor 'gradients/Add_1_grad/Reshape:0' shape=() dtype=float32>]
c_stop = tf.stop_gradient(c)

loss = tf.add(c_stop,b)  # 把c换成c_stop 之后，w1,w3,a,c的梯度都变为了None,
# [None, <tf.Tensor 'gradients/Mul_1_grad/Reshape:0' shape=() dtype=float32>, None, None, <tf.Tensor 'gradients/Add_1_grad/Reshape_1:0' shape=() dtype=float32>, None]
# 改变loss计算方式，把C换成1.0也有一样的结果
# 但是如果把b换成1.0，则只有b节点的梯度变为None
# [<tf.Tensor 'gradients/Mul_grad/Reshape:0' shape=() dtype=float32>, <tf.Tensor 'gradients/Mul_grad/Reshape_1:0' shape=() dtype=float32>, <tf.Tensor 'gradients/Add_grad/Reshape:0' shape=() dtype=float32>, <tf.Tensor 'gradients/Add_grad/Reshape_1:0' shape=() dtype=float32>, None, <tf.Tensor 'gradients/Add_1_grad/Reshape:0' shape=() dtype=float32>]

gradients = tf.gradients(loss,[w1,w2,w3,a,b,c])
print(gradients)
