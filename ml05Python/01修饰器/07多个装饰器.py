from time import time


def run_time_1(func):
    def wrap(a, b):
        r = func(a, b)
        print('run_time_1')
        return r
    return wrap

def run_time_2(func):
    def wrap(a, b):
        r = func(a, b)
        print('run_time_2')
        return r
    return wrap

@run_time_1
@run_time_2
def foo(a, b):
    return a + b

print(foo(2, 45))

""" 离被修饰函数最近的一侧先执行
run_time_2
run_time_1
47
"""
