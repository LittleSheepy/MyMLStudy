from functools import wraps


def namedDecorator(name):
    def run_time(func):
        @wraps(func)        # 使用@wraps装饰器可以消除这种影响
        def wraper(a, b):
            '''my decorator'''
            print('this is:{}'.format(name))
            r = func(a, b)
            return r

        return wraper

    return run_time

@namedDecorator("装饰器带参数")
def foo(a, b):
    """example docstring"""
    return a + b


print(foo(2, 45))
print(foo.__name__, foo.__doc__)

"""
this is:装饰器带参数
47
foo example docstring
"""
