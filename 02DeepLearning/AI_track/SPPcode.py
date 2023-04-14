import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPP(nn.Module):
    def __init__(self, output_size: list):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        b, _, h, w = x.shape
        outputs = [F.max_pool2d(x, kernel_size=(math.ceil(h / self.output_size[i]), math.ceil(w / self.output_size[i])),
                                stride=(math.floor(h / self.output_size[i]), math.floor(w / self.output_size[i]))).view(
            b, -1)
                   for i in range(len(self.output_size))]
        print([f'branch{i} output shape: {x.shape}' for i, x in enumerate(outputs)])
        return torch.cat(outputs, dim=1)

# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = math.floor((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
        w_pad = math.floor((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp

inputs = torch.randn(4, 256, 10, 20)
# print('--- classification config ---')
# spp = SPP(output_size=[4, 2, 1])
# outputs = spp(inputs)
# print(outputs.shape, '\n')
# print('--- object detection config ---')
# spp = SPP(output_size=[6, 3, 2, 1])
# outputs = spp(inputs)
# print(outputs.shape)
#
#
# sppl = SPPLayer([4, 2, 1])

# spatial_pyramid_pool

outputs = spatial_pyramid_pool(inputs, 4, [10, 20], [4, 2, 1])