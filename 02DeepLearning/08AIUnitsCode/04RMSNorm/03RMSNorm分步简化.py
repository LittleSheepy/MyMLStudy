import torch
from torch import nn
import numpy as np

seed = 996
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.manual_seed(seed)
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# shap 2 2 4
data = [[[1,2,3,4],
         [10,20,30,40]],

        [[1,2,3,4],
         [10,20,30,40]]]
data_np = np.array(data)

data_tensor = torch.Tensor(data)
bn = RMSNorm(4)
output = bn(data_tensor)
print("output=\n", output)
"""
output=
 tensor([[[0.3651, 0.7303, 1.0954, 1.4606],
         [0.3651, 0.7303, 1.0954, 1.4606]],
         
        [[0.3651, 0.7303, 1.0954, 1.4606],
         [0.3651, 0.7303, 1.0954, 1.4606]]], grad_fn=<MulBackward0>)
"""
bn_weight = bn.weight.data
bn_eps = bn.eps
print("bn.weight.data=\n", bn.weight.data)

# # 求均值
# shape = data_tensor.shape               # torch.Size([2, 2, 4])
# #mean = data_np.mean([0, 2, 3])
# mean = data_tensor.mean([2])        # tensor([[ 2.5000, 25.0000], [ 2.5000, 25.0000]])
# print("mean=\n", mean)
#
# # 求方差
# mean1 = mean.reshape(4, 1)                   # torch.Size([2, 1, 1])
# # mean1 = mean.unsqueeze(-1).unsqueeze(-1)      # torch.Size([2, 1, 1])
# print("mean1=\n", mean1)
# cha = data_tensor - mean1                       # torch.Size([2, 2, 2, 2])
# print("cha=\n", cha)

fang = torch.square(data_tensor)
print("fang=\n", fang)
rms = fang.mean([2])                                #  tensor([[  7.5000, 750.0000], [  7.5000, 750.0000]])
print("var=\n", rms)
# var_t = data_tensor.var([2], unbiased=False)        # tensor([  1.2500, 125.0000])
# print("var_t=\n", var_t)

# x_n = x/ sqrt(var + eps)
x_n = data_tensor / torch.sqrt(rms[..., None] + bn_eps)

# x1*gamma + beta
result = x_n * bn_weight
print("result=\n", result)


pass










