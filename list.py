# -*- coding: utf-8 -*-
# (C) Guangcai Ren <renguangcai@jiaaocap.com>
# All rights reserved
# create time '2019/6/26 14:41'
import math
import random
import time
from threading import Thread

_result_list = []


def split_df():
    # 线程列表
    thread_list = []
    # 需要处理的数据
    _l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # 每个线程处理的数据大小
    split_count = 2
    # 需要的线程个数
    times = math.ceil(len(_l) / split_count)
    count = 0
    for item in range(times):
        _list = _l[count: count + split_count]
        # 线程相关处理
        thread = Thread(target=work, args=(item, _list,))
        thread_list.append(thread)
        # 在子线程中运行任务
        thread.start()
        count += split_count

    # 线程同步，等待子线程结束任务，主线程再结束
    for _item in thread_list:
        _item.join()


def work(df, _list):
    """ 线程执行的任务，让程序随机sleep几秒

    :param df:
    :param _list:
    :return:
    """
    sleep_time = random.randint(1, 5)
    print(f'count is {df},sleep {sleep_time},list is {_list}')
    time.sleep(sleep_time)
    _result_list.append(df)


def use():
    split_df()


if __name__ == '__main__':
    y = use()
    print(len(_result_list), _result_list)