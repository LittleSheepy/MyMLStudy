import torch
from torch import nn

seed = 996
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.manual_seed(seed)


class bn_test(nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels out_channels kernel_size stride padding
        # self.mbn = nn.Conv2d(1, 3, 3, 2, bias=True)
        self.con = nn.Conv2d(1, 3, 3, 2, bias=False)
        self.bn = nn.BatchNorm2d(3)

        self.bn_weight = None
        self.bn_bias = None
        self.con_ret = None

    def weights_init(self):
        self.bn_weight = self.bn.weight.data
        self.bn_bias = self.bn.bias.data

    def forward(self, x):
        n1 = self.con(x)    # 2 3 2 2
        self.con_ret = n1  # 记录con2d结果用于后面手工计算BN结果
        n2 = self.bn(n1)
        return n2


m = bn_test()
m.weights_init()
data = torch.rand((2, 1, 5, 5))
ret = m(data)  # 得到torch BN结果

#
#
# 我们手动计算BN的结果和torch BatchNorm2d做对比是否正确
#
#

# 求均值
bn_mean = m.con_ret.mean([0, 2, 3])
print("con_ret=\n", m.con_ret)
print("bn_mean=\n", bn_mean)
# 这里需要注意，torch.var 默认算无偏估计，因此需要手动设置unbiased=False
bn_var = m.con_ret.var([0, 2, 3], unbiased=False)
print("bn_var=\n", bn_var)

# x1 = (x - mean) / sqrt(var + eps)
x1 = (m.con_ret - bn_mean[None, ..., None, None]) / torch.sqrt(bn_var[None, ..., None, None] + m.bn.eps)
# x1*gamma + beta
bn_result = x1 * m.bn_weight[..., None, None] + m.bn_bias[..., None, None]

# 打印看下是否结果相等
print(ret)
print(bn_result)