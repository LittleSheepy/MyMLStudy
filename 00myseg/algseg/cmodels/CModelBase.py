from algseg.config.CCfgBase import CCfgBase


class CModelBase:
    def __init__(self, config: CCfgBase):
        self.config = config
        self.callback_print = None
        self.train_lossList = []


    def close(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def train(self):
        pass

    def save(self):
        pass

    def test(self):
        pass

    def set_callback_print(self, func):
        self.callback_print = func

    def print_loss(self, str):
        if self.callback_print:
            self.callback_print(str)
        else:
            # print(str)
            pass

    def get_data_list(self, data_name):
        if "train_loss" == data_name:
            return self.train_lossList

    def put_data_list(self, data_name, data):
        if "train_loss" == data_name:
            self.train_lossList.append(data)