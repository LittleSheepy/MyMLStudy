import math
import time
from contextlib import ContextDecorator


# 自定义一个装饰器
class timer(ContextDecorator):
    def __enter__(self):
        self.start = time.time()
        self.t = 0

    def __exit__(self, *exc):
        self.t = time.time() - self.start
        print('the function spend  time is', self.t)


# 函数1
@timer()
def create_list(x):
    return [i for i in range(x)]


# 函数2
@timer()
def cal_sin(x):
    return [math.sin(x) for i in range(x)]

if __name__ == '__main__':
    print("")
    # 开始测试
    create_list(100 * 10000)
    cal_sin(100 * 1000)
    cal_sin(100 * 10000)

    # 可以 用 with 语句
    t = timer()
    with t:
        cal_sin(100 * 100000)
    print(t.t)


