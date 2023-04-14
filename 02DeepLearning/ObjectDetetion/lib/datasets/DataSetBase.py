import numpy as np

class DataSetBase():
    def __init__(self):
        self.cursor = 0         # 当前开始读取数据的光标
        self.epoch = 0          # 批次
        self.cls_names = None
        self._cls_num = None
        self._cls_to_id = None

    def next_batch(self, batch_size):
        endPos = self.cursor + batch_size
        if endPos >= self.num_label:
            result = self.labels[self.cursor:]
            self.cursor = 0
            np.random.shuffle(self.labels)
        else:
            result = self.labels[self.cursor:endPos]
            self.cursor = endPos
        return result

    @property
    def cls_to_ind(self):
        if self._cls_to_id is None:
            self._cls_to_id = dict(zip(self.cls_names, range(self.cls_num)))
        return self._cls_to_id

    @property
    def cls_num(self):
        if self._cls_num is None:
            self._cls_num = len(self.cls_names)
        return self._cls_num