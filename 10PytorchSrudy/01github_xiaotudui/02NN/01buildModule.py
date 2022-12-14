import torch
from torch import nn

# 搭建网络
class myModule(nn.Module):
    def __init__(self):
        super(myModule, self).__init__()
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
    def forward(self, x):
        x = self.conv(x)
        return x

if __name__ == '__main__':
    my_module = myModule()
    input = torch.ones((2, 1, 8, 8))
    output = my_module(input)
    print(output.shape)