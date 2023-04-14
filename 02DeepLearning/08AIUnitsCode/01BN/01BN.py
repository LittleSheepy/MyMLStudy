import torch
from torch import nn
class myBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(myBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha                                # class中新加的一个参数alpha

    def forward(self, input):                             # input(N,C,H,W)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:    # training和track_running_stats都为True才更新BN的参数
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1             # 记录完成前向传播的batch数目
                if self.momentum is None:                 # momentum为None，用1/num_batches_tracked代替
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:                                     # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates 计算均值和方差的过程
        if self.training:
            mean = input.mean([0, 2, 3])                  #计算均值，出来的维度大小等于channel的数目
            # torch.var 默认算无偏估计，这里先算有偏的，因此需要手动设置unbiased=False
            var = input.var([0, 2, 3], unbiased=False)    # 计算的是有偏方差
            n = input.numel() / input.size(1)             # size(1)是指channel的数目  n=N*H*W
            with torch.no_grad():                         # 计算均值和方差的过程不需要梯度传输
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var 这里通过有偏方差和无偏方差的关系，又转换成了无偏方差
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:                                             # 不处于训练模式就固定running_mean和running_var的值
            mean = self.running_mean
            var = self.running_var
        # 用None扩充维度，然后与原输入tensor做相应运算实现规范化

        # 报错
        # input = self.alpha * ((input - mean[None, ..., None, None])\(torch.sqrt(var[None, ..., None, None] + self.eps)))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input