import tensorflow as tf
import numpy as np

print(tf.__version__)
b=[1,2,3]
b=tf.convert_to_tensor(b,dtype=float)
print(b)
b1=tf.expand_dims(b,axis=0)
print("b1", b1)
b1=tf.tile(b1,multiples=[2,1]) #multiples列表表示在对应维度上复制的倍数（1倍为不复制）
b2=tf.expand_dims(b,axis=1)
print("b2", b2)
b2=tf.expand_dims(b,axis=1)
print("b21", b2)
b2=tf.tile(b2,multiples=[1,3])
print(b1)
print(b2)
import  tensorflow as tf
from tensorflow.keras import layers,optimizers,datasets

x=tf.range(4)
x=tf.reshape(x,[2,2])
print(x)
x=tf.tile(x,multiples=[1,2])
y=tf.tile(x,multiples=[2,1])
print("x")
print(x)
print("y")
print(y)




