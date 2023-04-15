# -*- coding:utf-8 -*-
"""
作者：wyt
日期：2022年04月21日
"""
import threading
import requests
import time

urls = [
    f'https://www.cnblogs.com/#p{page}'  # 待爬地址
    for page in range(1, 10)  # 爬取1-10页
]


def craw(url):
    r = requests.get(url)
    num = len(r.text)  # 爬取博客园当页的文字数
    return num  # 返回当页文字数


def sigle():  # 单线程
    res = []
    for i in urls:
        res.append(craw(i))
    return res


class MyThread(threading.Thread):  # 重写threading.Thread类，加入获取返回值的函数
    def __init__(self, url):
        threading.Thread.__init__(self)
        self.url = url  # 初始化传入的url

    def run(self):  # 新加入的函数，该函数目的：
        self.result = craw(self.url)  # ①。调craw(arg)函数，并将初试化的url以参数传递——实现爬虫功能
        # ②。并获取craw(arg)函数的返回值存入本类的定义的值result中

    def get_result(self):  # 新加入函数，该函数目的：返回run()函数得到的result
        return self.result


def multi_thread():
    print("start")
    threads = []  # 定义一个线程组
    for url in urls:
        threads.append(  # 线程组中加入赋值后的MyThread类
            MyThread(url)  # 将每一个url传到重写的MyThread类中
        )
    for thread in threads:  # 每个线程组start
        thread.start()
    for thread in threads:  # 每个线程组join
        thread.join()
    list = []
    for thread in threads:
        list.append(thread.get_result())  # 每个线程返回结果(result)加入列表中
    print("end")
    return list  # 返回多线程返回的结果组成的列表


if __name__ == '__main__':
    start_time = time.time()
    result_multi = multi_thread()
    print(result_multi)  # 输出返回值-列表
    end_time = time.time()
    print('用时：', end_time - start_time)
    start_time = time.time()
    result_sig = sigle()
    print(result_sig)
    end_time = time.time()
    print('用时：', end_time - start_time)
