from algseg.cmodels.CModelBase import CModelBase
from algseg.config.CCfgExample import CCfgExample
import time


class CExample(CModelBase):
    def __init__(self, config: CCfgExample):
        super().__init__(config)
        self.config = config

    def train(self):
        epochs = self.config.train_epoches
        for epoch in range(1, epochs + 1):
            time.sleep(1)
            loss = ((100-epoch)**2)*0.0001
            loss = round(loss, 8)
            str_loss = "loss:{}".format(loss)
            self.put_data_list("train_loss", str_loss)
            self.print_loss(str_loss)


