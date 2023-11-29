import threading
import time


class CLS:
    def __init__(self):
        self.train_flg = 1

    def train(self):
        while True:
            time.sleep(1)
            print("running")
            if self.train_flg == 2:
                print("break!!!")
                break
            if self.train_flg == 3:
                while True:
                    if self.train_flg != 3:
                        break
                    time.sleep(2)
                    print("暂停ing")

    def set(self, flg):
        self.train_flg = flg
cls = CLS()
t1 = threading.Thread(target=cls.train)
print("")
print("start")
t1.start()

time.sleep(5)
cls.set(3)
time.sleep(5)
cls.set(1)
time.sleep(10)
cls.set(2)
time.sleep(5)
print("join")
t1.join()
print("finished")