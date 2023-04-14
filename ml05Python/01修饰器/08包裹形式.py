from time import time


def run_time_1(func):
    def wrap(a, b):
        """run_time_1 docstring"""
        print("     第二层 ", func.__name__, func.__doc__)
        r = func(a, b)
        print("     第二层 ", func.__name__, func.__doc__)
        print('run_time_1')
        return r
    return wrap

def run_time_2(func):
    def wrap(a, b):
        """run_time_2 docstring"""
        r = func(a, b)
        print('     run_time_2')
        return r
    return wrap

@run_time_1
@run_time_2
def foo(a, b):
    """foo docstring"""
    return a + b

print("最外层 ", foo.__name__, foo.__doc__)
foo(2, 45)
print("最外层 ", foo.__name__, foo.__doc__)

""" 就像剥洋葱一样
最外层  wrap run_time_1 docstring
     第二层  wrap run_time_2 docstring
     run_time_2
     第二层  wrap run_time_2 docstring
run_time_1
最外层  wrap run_time_1 docstring
"""
