from algseg.config.CCfgBase import CCfgBase


class CModelBase:
    def __init__(self, config: CCfgBase):
        self.config = config
        self.callback_print = None

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
            print(str)

