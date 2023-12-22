import myPack2
"""
attempted relative import beyond top-level package
尝试在顶级包之外进行相对导入
"""
# from dir2.dir2f1 import *
from myPack2.dir2.dir2f1 import *
if __name__ == '__main__':
    print("")
    pk2dir2f1fun1()
