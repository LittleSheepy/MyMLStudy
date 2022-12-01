import torch
from torch import nn
import numpy as np

seed = 996
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.manual_seed(seed)


# shap 2 1 2 2
data = [[[[1,2],[3,4]],[[10,20],[30,40]]],
        [[[1,2],[3,4]],[[10,20],[30,40]]]]
data_np = np.array(data)

data_tensor = torch.Tensor(data)
bn = nn.BatchNorm2d(2)
output = bn(data_tensor)
print("output=\n", output)
bn_weight = bn.weight.data
bn_bias = bn.bias.data
bn_eps = bn.eps
print("bn.weight.data=\n", bn.weight.data)
print("bn.bias.data=\n", bn.bias.data)

# 求均值
shape = data_tensor.shape
#mean = data_np.mean([0, 2, 3])
mean = data_tensor.mean([0,2,3])
print("mean=\n", mean)

# 求方差
mean1 = mean.unsqueeze(-1)
mean1 = mean1.unsqueeze(-1)
print("mean1=\n", mean1)
cha = data_tensor - mean1
print("cha=\n", cha)

fang = torch.square(cha)
print("fang=\n", fang)
var = fang.mean([0,2,3])
print("var=\n", var)
var_t = data_tensor.var([0,2,3], unbiased=False)
print("var_t=\n", var_t)

# x_n = (x - mean) / sqrt(var + eps)
x_n = (data_tensor - mean[None, ..., None, None]) / torch.sqrt(var[None, ..., None, None] + bn_eps)

# x1*gamma + beta
result = x_n * bn_weight[..., None, None] + bn_bias[..., None, None]
print("result=\n", result)













