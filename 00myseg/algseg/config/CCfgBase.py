from pathlib import Path
"""
train_batch_size, 数字
train_epoches, 数字
model_type, ["seg", "det", "cls"]
model_size, ["large", "middle", "small"]
model_name, string
output_dir, string

"""
class CCfgBase():
    def __init__(self, **para):
        # 数据集
        self.data_root = r"./data/"
        self.data_name = "chase_db1"
        self.data_classes = 2
        # 模型
        self.model_name = "unet"
        self.model_size = "small"
        # 训练
        self.train_lr = 0.001
        self.train_epoches = 100
        self.train_batch_size = 10
        # 保存
        self.output_dir = Path('./checkpoints/')
    # def check_para(self):
    #     para_require = ["batch_size", "train_epoch"]
    #     for key, value in para.items():
    #         setattr(self, key, value)

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s=%s' % (key, attrs[key]) for key in attrs]
        return ', '.join(result)

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (int, float, bool, str):
                result[attr] = value
        return result

    def set_paras(self, **para):
        for key, value in para.items():
            setattr(self, key, value)


if __name__ == '__main__':
    cfg = CCfgBase()
    print('-' * 100)
    print(cfg)
    cfg.set_paras(model_name="unet++")
    print(cfg)
    cfg.set_paras(model_name="unet", model_size="large")
    print(cfg)