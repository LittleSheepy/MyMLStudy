"""
方法解析顺序（Method Resolution Order），简称 MRO
实际上，Python 发展至今，经历了以下 3 种 MRO 算法，分别是：

1.从左往右，采用深度优先搜索（DFS）的算法，称为旧式类的 MRO；
2.自 Python 2.2 版本开始，新式类在采用深度优先搜索算法的基础上，对其做了优化；

3.自 Python 2.3 版本，对新式类采用了 C3 算法。
由于 Python 3.x 仅支持新式类，所以该版本只使用 C3 算法。


"""


class FatherBase:
    def __init__(self):
        print(">> FatherBase:__init__")
        self.name = "FatherBase"
        print("set name FatherBase")
        print("<<<FatherBase:__init__")

class FatherBaseA:
    def __init__(self):
        print(">> FatherBaseA:__init__")
        self.name = "FatherBaseA"
        print("set name FatherBaseA")
        print("<<<FatherBaseA:__init__")



class FatherA(FatherBase):
    def __init__(self):
        print(">> FatherA:__init__")
        super().__init__()
        self.name = "FatherA"
        print("set name FatherA")
        print("<<<FatherA:__init__")

class FatherB(FatherBase):
    def __init__(self):
        print(">> FatherB:__init__")
        super().__init__()
        self.name = "FatherB"
        print("set name FatherB")
        print("<<<FatherB:__init__")

class Child(FatherA, FatherB):
    def __init__(self):
        print(">> Child:__init__")
        super().__init__()
        print("<<<Child:__init__")

if __name__ == '__main__':
    print("")
    child = Child()
    print(Child.mro())
    """
    [<class '__main__.Child'>, 
    <class '__main__.FatherA'>, 
    <class '__main__.FatherBaseA'>, 
    <class '__main__.FatherB'>, 
    <class '__main__.FatherBase'>, 
    <class 'object'>]
    """
    # print(child.mothod())
    # print(child.name)