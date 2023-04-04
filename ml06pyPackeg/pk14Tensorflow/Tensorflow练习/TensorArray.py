import tensorflow as tf
import numpy as np

sess = tf.Session()
x = np.arange(20)
input_ta = tf.TensorArray(size=0, dtype=tf.int32, dynamic_size=True)
input_ta = input_ta.unstack(x)  # TensorArray可以传入array或者tensor
for time in range(len(x)):
    print(sess.run(input_ta.read(time)))  # 遍历查看元素

output = input_ta.stack()  # 合成
print(sess.run(output))

for time in range(5):
    input_ta = input_ta.write(time + len(x), time)  # 写入

output = input_ta.stack()
print(sess.run(output))
