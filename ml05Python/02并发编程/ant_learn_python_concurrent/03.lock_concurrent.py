import threading
from threading import Thread
import time

lock = threading.Lock()

class Account:
    def __init__(self, balance):
        self.balance = balance

def draw(account:Account, amoumt):
    name = threading.current_thread().name
    with lock:
        if account.balance >= amoumt:
            time.sleep(0.1)
            print(name, "取钱成功")
            account.balance -= amoumt
            print(name, "余额", account.balance)
        else:
            print(name, "取钱失败，余额不足")

if __name__ == '__main__':
    account = Account(1000)
    ta = Thread(name='ta', target=draw, args=(account, 800))
    tb = Thread(name='tb', target=draw, args=(account, 800))

    ta.start()
    tb.start()

