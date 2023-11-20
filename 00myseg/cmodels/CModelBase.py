from config.CCfgBase import CCfgBase
class CModelBase:
    def __init__(self, config: CCfgBase):
        self.config = config

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
