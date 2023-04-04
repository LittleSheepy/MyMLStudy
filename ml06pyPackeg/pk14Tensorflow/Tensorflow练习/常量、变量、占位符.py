import tensorflow.compat.v1 as tf
print(tf.__version__)

"""
def constant_v1(
    value, 
    dtype=None,  张量类型
    shape=None,  
    name="Const", verify_shape=False):
"""
print("---constant---"*10)
c1 = tf.constant(3)
c1_3 = 3*c1
c2 = tf.constant(3,tf.float32)
c3 = tf.constant(3,shape=[2,3])
c4 = tf.constant([1,1,1],shape=[2,3])
c9 = tf.constant(9, tf.float32, [], "c9", False)
c11 = tf.random_normal([3,1,2])
c12 = tf.random_normal([3,2])
with tf.Session() as session:
    print(c1_3)
    print("c1 = ",session.run(c1),type(session.run(c1)))
    print("c2 = ",session.run(c2))
    print("c3 = ",session.run(c3))
    print("c4 = ",session.run(c4))
    print(f"c9: value={session.run(c9)}; c9.dtype={c9.dtype};c9.shape={c9.shape},c9.name={c9.name} c9.eval()={c9.eval()}")
    print(session.run(c11+c12))

"""
def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=None,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 use_resource=None,
                 custom_getter=None,
                 constraint=None,
                 synchronization=VariableSynchronization.AUTO,
                 aggregation=VariableAggregation.NONE):
"""
print("---get_variable---"*10)
v = tf.get_variable("v", [], tf.float32, tf.initializers.ones)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(v))
with tf.Session() as session:
    print(session.run(v,{v:22}))


"""
占位符
def placeholder(dtype, shape=None, name=None):
"""
print("---placeholder---"*10)
x = tf.placeholder(tf.float32, [], name="x")
print("x.shape",x.shape)
with tf.Session() as session:
    print(session.run(x,{x:33}))


