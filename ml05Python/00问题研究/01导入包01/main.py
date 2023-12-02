"""
描述：
行1加上就报错，去掉就能正常运行。行1在行2前后都报错。

原因：
未确定
猜测：
行1用from的时候，myPack.dir 下的 f1 会看成一个包或者块，
然后最后把 f1 的类型由 《函数类型》替换为《module类型》。
这样，getf1()获取的 f1 就变成了一个module

避免方式:
1.文件名和文件内的函数名、类名、变量名 不一样即可避免
2.采用别名方式
myPack.dir.__init__ 文件多的
from dir.f1 import f1
def getf1():
    return f1
改为
from dir.f1 import f1 as f
def getf1():
    return f
"""
from myPack.dir.f1 import f1     # 行1 加上就报错
from myPack.dir import getf1              # 行2

if __name__ == '__main__':
    print(type(getf1()))
    getf1()()

