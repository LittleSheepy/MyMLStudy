

class CfgBase:
    def __init__(self):
        self.name = "CfgBase"

    def func_self(self):
        print("CfgBase func_self")

    def func_call(self):
        self.func_self()

class CfgA(CfgBase):
    def __init__(self):
        super().__init__()
        self.name = "CfgA"
        self.paraA = "A"

    def func_self(self):
        print("CfgA func_self")


class MyCls:
    def __init__(self, cfg: CfgBase):
        print(cfg.paraA)

if __name__ == '__main__':
    print("")
    cfg = CfgA()
    # mycls = MyCls(cfg)
    cfg.func_call()