from time import time

def run_time(func):
    def wrap(a, b):
        t1 = time()
        func(a, b)
        t2 = time()
        print(t2 - t1)
    return wrap

@run_time
def foo(a, b):
    print(a + b)

foo(2, 45)

"""
47
0.0
"""
