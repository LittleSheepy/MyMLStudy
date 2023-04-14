"""
y = x**2
delt_y = x**2 - a
dy = 2*x = delt_y/delt_x
delt_x = delt_y / dy = (x**2 - a)/(2*x)
x = x - delt_x = (a + x*x) / (2 * x)
"""
# 牛顿法
def sqrt_newton(a):
    x = 1
    for i in range(2000):
        x = (a + x*x) / (2 * x)
    return x

# 二分法
def sqrt_binary(a, eps=1e-4):
    xa = 0
    xb = (a+1)**2
    while abs(xa - xb) > eps:
        xm = (xa + xb)/2
        if xm * xm >= a:
            xb = xm
        else:
            xa = xm
    return xa

"""
y = x**2
delt_y = x**2 - a
dy = 2*x
delt_x = delt_y * dy * lr
"""
# GD1
def sqrt_GD1(a, lr=0.001):
    y = lambda x:x**2
    dy = lambda x:2*x
    delt_y = lambda x: a - y(x)
    delt_x = lambda x, lr:delt_y(x) * dy(x) * lr
    x = 1
    for i in range(2000):
        x += delt_x(x, lr)
    return x

"""
y = x**2
loss = (y - a)**2
dloss = 2*(y-a)*2*x
delt_x = -dloss * lr
"""
# GD2
def sqrt_GD2(a, lr=0.001):
    y = lambda x:x**2
    loss = lambda x:(a - y(x))**2
    dloss = lambda x:2*(a - y(x))*(-2*x)
    delt_x = lambda x:-dloss(x)*lr

    x = 1
    for i in range(2000):
        x += delt_x(x)
    return x

import tensorflow as tf
def sqrt_GD2_tf(a_v, lr_v=0.001, epoches=2000):
    x = tf.get_variable("v", [], tf.float32)
    a = tf.placeholder(tf.float32, [], "n")
    lr = tf.placeholder(tf.float32, [], "lr")
    loss = tf.square(tf.square(x)-a)
    opt = tf.train.GradientDescentOptimizer(lr)
    train_op = opt.minimize(loss)
    x = tf.abs(x)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for _ in range(epoches):
            session.run(train_op, {a:a_v, lr:lr_v})
        return session.run(x)


def two_x(lr=0.01):
    y = lambda x1, x2:(x1-3)**2 + (x2+4)**2
    dy_dx1 = lambda x:2*(x-3)
    dy_dx2 = lambda x:2*(x+4)
    delt_x1 = lambda x:-lr*dy_dx1(x)
    delt_x2 = lambda x:-lr*dy_dx2(x)

    x1 = 1
    x2 = 1
    for i in range(2000):
        x1 += delt_x1(x1)
        x2 += delt_x2(x2)
    return x1, x2




if __name__ == '__main__':
    print('squrt_newton(2) = %f' % sqrt_newton(2))
    print('sqrt_binary(2) = %f' % sqrt_binary(2))
    print('squrt_GD1(2) = %f' % sqrt_GD1(2))
    print('squrt_GD2(2) = %f' % sqrt_GD2(2))
    print('sqrt_GD2_tf(2) = %f' % sqrt_GD2_tf(2))
    print('x1, x2 =',two_x())