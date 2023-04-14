from time import time

def run_time(func):
    def wrap(a, b):
        t1 = time()
        r=func(a, b)
        t2 = time()
        print(t2 - t1)
        return r
    return wrap

@run_time
def foo(a, b):
    return a + b

print(foo(2, 45))

"""
0.0
47
"""
