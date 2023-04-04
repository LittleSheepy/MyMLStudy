import tensorflow.compat.v1 as tf
import numpy as np

print(tf.__version__)

print("---tf.nn.l2_normalize---"*5)
"""
def l2_normalize(x, 
    axis=None,     规范化的维度.标量或整数向量.
    epsilon=1e-12, 
    name=None, 
    dim=None):     axis的不推荐别名
"""
a = tf.constant([[1,2,3],[4,5,6],[7,8,9]], dtype=tf.float32)
"""
axis=0
norm(1) = L2(1,4,7) = (1+16+49)**0.5
norm(2) = L2(2,5,8)
norm(3) = L2(3,6,9)
[[1./norm(1), 2./norm(2) , 3./norm(3) ]
[4./norm(1) , 5./norm(2) , 6./norm(3) ]    =
[7./norm(1) , 8./norm(2) , 9./norm(3) ]]
[[0.12309149 0.20739034 0.26726127]
[0.49236596 0.51847583 0.53452253]
[0.86164045 0.82956135 0.80178374]]
"""
norL2_a0 = tf.nn.l2_normalize(a, axis=0)
"""
axis=1
norm(1) = L2(1,2,3) = (1+4+9)**0.5
norm(2) = L2(4,5,6)
norm(3) = L2(7,8,9)
[[1./norm(1), 2./norm(1) , 3./norm(1) ]
[4./norm(2) , 5./norm(2) , 6./norm(2) ]    =
[7./norm(3) , 8..norm(3) , 9./norm(3) ]]
[[0.26726124 0.5345225  0.8017837 ]
[0.45584232 0.5698029  0.6837635 ]
[0.5025707  0.5743665  0.64616233]]
"""
norL2_a1 = tf.nn.l2_normalize(a, axis=1)

with tf.Session() as session:
    norL2_a0_v = session.run(norL2_a0)
    print("norL2_a0_v\n",norL2_a0_v)
    print(1/(1+16+49)**0.5)

    norL2_a1_v = session.run(norL2_a1)
    print("norL2_a0_v\n",norL2_a1_v)
    print(1/(1+4+9)**0.5)


print("---tf.norm---"*10)
"""
def norm(tensor,
         ord='euclidean',
         axis=None,
         keepdims=None,
         name=None,
         keep_dims=None):
"""
a = tf.constant([1,2,3,4],dtype=tf.float32)
b = tf.constant([[1,2],[3,4]],dtype=tf.float32)
"""
ord:
    euclidean:欧几里得  sqrt(a**2+b**2)
    1: a+b+c
    2: L2 范数
"""
norm1 = tf.norm(a, ord="euclidean", axis=0)
norm2 = tf.norm(a, ord=1, axis=0)
norm3 = tf.norm(a, ord=2, axis=0)
norm4 = tf.norm(b, ord=2)
norm5 = tf.norm(b, ord=1, axis=0)
norm6 = tf.norm(b, ord=1, axis=1)
with tf.Session() as session:
    norm1_v = session.run(norm1)
    print("norm1_v",norm1_v)
    print(np.sqrt(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4))
    norm2_v = session.run(norm2)
    print("norm2_v",norm2_v)
    norm3_v = session.run(norm3)
    print("norm3_v",norm3_v)
    norm4_v = session.run(norm4)
    print("norm4_v",norm4_v)
    norm5_v = session.run(norm5)
    print("norm5_v",norm5_v)
    norm6_v = session.run(norm6)
    print("norm6_v",norm6_v)




