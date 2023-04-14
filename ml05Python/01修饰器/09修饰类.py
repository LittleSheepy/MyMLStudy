def decorator(cls):
    print(">>>AAA 这里可以写被装饰类新增的功能")
    return cls

@decorator
class A(object):
    def __init__(self):
        pass

    def test(self):
        print("test")

def decoratorB(cls):
    print(">>>BBB 这里可以写被装饰类新增的功能")
    class B_new(cls):
        def __init__(self):
            print("B_new")
            super().__init__()
    return B_new

@decoratorB
class B(object):
    def __init__(self):
        pass

    def test(self):
        print("test")
"""
>>>这里可以写被装饰类新增的功能
"""
a = A()
b = B()
