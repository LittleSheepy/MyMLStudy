def foo_no(a, b):
    """foo_no example docstring"""
    return a + b
def namedDecorator(name):
    def run_time(func):
        def wrap(a, b):
            '''my decorator'''
            print('this is:{}'.format(name))
            r = func(a, b)
            return r
        return wrap
    return run_time

@namedDecorator("装饰器带参数")
def foo(a, b):
    """foo example docstring"""
    return a + b

print(foo(2, 45))
print(foo.__name__, ",", foo.__doc__)
print(foo_no.__name__, ",", foo_no.__doc__)

"""
this is:装饰器带参数
47
wrap , my decorator
foo_no , foo_no example docstring
"""
