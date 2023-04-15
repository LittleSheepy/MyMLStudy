# 创建一个多线程编程的例子
import threading

def print_numbers(num):
    for i in range(1, num):
        print(i)

def print_letters(str):
    for letter in str*10:
        print(letter)

# 创建两个线程
t1 = threading.Thread(target=print_numbers,args=(100,))
t2 = threading.Thread(target=print_letters, args=("abcdefghij",))

# 启动线程
t1.start()
t2.start()

# 等待线程结束
t1.join()
t2.join()