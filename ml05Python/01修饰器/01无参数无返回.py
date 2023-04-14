from time import time


def run_time(func):
    print(">>> 修饰时候 执行且只执行一次修饰函数")
    def wrap():
        t1 = time()
        func()
        t2 = time()
        print(t2 - t1)
    return wrap

@run_time
def foo():
    print('hello')

@run_time
def foo2():
    print('hello')

if __name__ == '__main__':
    foo()