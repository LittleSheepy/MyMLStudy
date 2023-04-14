from time import time

def run_time(func):
    def wrap():
        t1 = time()
        r = func()
        t2 = time()
        print(t2-t1)
        return r
    return wrap

@run_time
def foo():
    return 'hello'

print(foo())
