import tensorflow as tf
print(tf.__version__)

"""
reduce_mean,reduce_min,reduce_max, reduce_sum,
tf.reduce_all : 计算 tensor 指定轴方向上的各个元素的逻辑和（and 运算）;
tf.reduce_any: 计算 tensor 指定轴方向上的各个元素的逻辑或（or 运算）;
"""
print("---reduce_XXX---"*10)
g = tf.Graph()
with g.as_default():
    a = tf.random_uniform([2, 3, 4], 10.0, 10.0)
    mean = tf.reduce_mean(a)
    min = tf.reduce_min(a)
    max = tf.reduce_max(a)
    sum = tf.reduce_sum(a)

with tf.Session(graph=g) as session:
    a_v, mean_v, min_v, max_v,sum_v = session.run([a, mean, min, max, sum])
print("a_v",a_v)
print("mean, min, max, sum= ",mean_v, min_v, max_v, sum_v)

"""
def reduce_sum_v1(input_tensor,
                  axis=None,
                  keepdims=None,
                  name=None,
                  reduction_indices=None,
                  keep_dims=None):
"""
print("---reduce_sum---"*10)
with g.as_default():
    sum1 = tf.reduce_sum(a, 0)
    sum2 = tf.reduce_sum(a, 1)
    sum3 = tf.reduce_sum(a, 2)
with tf.Session(graph=g) as session:
    sum1_v,sum2_v,sum3_v = session.run([sum1, sum2, sum3])

print("sum1", sum1_v)
print("sum2", sum2_v)
print("sum3", sum3_v)



