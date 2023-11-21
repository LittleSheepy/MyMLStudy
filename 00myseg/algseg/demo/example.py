from algseg.cmodels.CExample import CExample
from algseg.config import get_cfg
import time


# 打印消息的回调函数
def callback_print(str):
    timestamp = time.time()
    date_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    print(date_string + ":  " + str)


if __name__ == '__main__':
    cfg = get_cfg("example")
    cfg.train_epoches = 10
    print(cfg)
    model = CExample(cfg)
    model.set_callback_print(callback_print)        # 设置打印回调函数
    model.train()

    # 获取train_loss列表 以供显示曲线。通过多线程可以在training时候获取
    train_loss_list =model.get_data_list("train_loss")
    print(train_loss_list)

