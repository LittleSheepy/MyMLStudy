def namedDecorator(name):
    def run_time1(func):
        def wrap(a, b):
            print('this is:{}'.format(name))
            r = func(a, b)
            return r
        return wrap
    return run_time1

@namedDecorator("装饰器带参数")
def foo(a, b):
    return a + b

print(foo(2, 45))

"""
this is:装饰器带参数
47
"""
