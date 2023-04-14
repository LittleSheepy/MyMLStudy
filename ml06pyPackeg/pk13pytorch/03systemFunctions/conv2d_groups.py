import torch
from torch import nn
from torch.nn.parameter import Parameter

# 分组卷积 groups
# 首先需要知道的是groups的值必须能整除in_channels和out_channels
# 卷积参数量的计算公式是：（输入通道数 * 输出通道数 * k^2 ）/ groups
x = torch.ones([1,6,18,18])
conv = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, groups=1)
w_size = conv.weight.data.size()        # torch.Size([3, 6, 1, 1])
y = conv(x)     # torch.Size([1, 3, 18, 18])
print(w_size, y.size())

conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=1)
w_size = conv.weight.data.size()        # torch.Size([6, 6, 1, 1])
y = conv(x)     # torch.Size([1, 6, 18, 18])
print(w_size, y.size())

conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=2)
w_size = conv.weight.data.size()        # torch.Size([6, 3, 1, 1])
conv.weight = Parameter(torch.ones_like(conv.weight.data))
conv.bias = Parameter(torch.zeros_like(conv.bias.data))
y = conv(x)     # torch.Size([1, 6, 18, 18]) 3
print(w_size, y.size())

conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=3)
w_size = conv.weight.data.size()        # torch.Size([6, 2, 1, 1])
conv.weight = Parameter(torch.ones_like(conv.weight.data))
conv.bias = Parameter(torch.zeros_like(conv.bias.data))
y = conv(x)     # torch.Size([1, 6, 18, 18]) 2
print(w_size, y.size())

conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=6)
w_size = conv.weight.data.size()        # torch.Size([6, 1, 1, 1])
y = conv(x)     # torch.Size([1, 6, 18, 18])
print(w_size, y.size())







