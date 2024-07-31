import torch
from torch import nn
import numpy as np

seed = 996
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.manual_seed(seed)


# shap 2 2 2 2
data = [[[[1,2],[3,4]],
         [[10,20],[30,40]]],

        [[[1,2],[3,4]],
         [[10,20],[30,40]]]]
data_np = np.array(data)

data_tensor = torch.Tensor(data)
bn = nn.LayerNorm([2,2,2])
output = bn(data_tensor)
print("output=\n", output)
"""
output=
 tensor([[[[-0.9257, -0.8531],
          [-0.7805, -0.7079]],
         [[-0.2723,  0.4538],
          [ 1.1799,  1.9059]]],
          
        [[[-0.9257, -0.8531],
          [-0.7805, -0.7079]],
         [[-0.2723,  0.4538],
          [ 1.1799,  1.9059]]]], grad_fn=<NativeLayerNormBackward0>)
"""
bn_weight = bn.weight.data
bn_bias = bn.bias.data
bn_eps = bn.eps
print("bn.weight.data=\n", bn.weight.data)
print("bn.bias.data=\n", bn.bias.data)

# 求均值
shape = data_tensor.shape               # torch.Size([2, 2, 2, 2])
#mean = data_np.mean([0, 2, 3])
mean = data_tensor.mean([1,2,3])        # tensor([13.7500, 13.7500])
print("mean=\n", mean)

# 求方差
mean1 = mean.reshape(2, 1, 1)                   # torch.Size([2, 1, 1])
# mean1 = mean.unsqueeze(-1).unsqueeze(-1)      # torch.Size([2, 1, 1])
print("mean1=\n", mean1)
cha = data_tensor - mean1                       # torch.Size([2, 2, 2, 2])
print("cha=\n", cha)

fang = torch.square(cha)
print("fang=\n", fang)
var = fang.mean([1,2,3])                                # tensor([  1.2500, 125.0000])
print("var=\n", var)
var_t = data_tensor.var([1,2,3], unbiased=False)        # tensor([  1.2500, 125.0000])
print("var_t=\n", var_t)

# x_n = (x - mean) / sqrt(var + eps)
x_n = (data_tensor - mean[None, ..., None, None]) / torch.sqrt(var[None, ..., None, None] + bn_eps)

# x1*gamma + beta
result = x_n * bn_weight[..., None, None] + bn_bias[..., None, None]
print("result=\n", result)


pass










