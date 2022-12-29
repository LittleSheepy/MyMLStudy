import torch
import torch.nn as nn
import os

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=1,bias=True)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=1,bias=True)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        # 用int函数求值
        batch = int(x.size(0))
        x = x.view(batch, -1)
        return x


# 这个包对应opset11的导出代码，如果想修改导出的细节，可以在这里修改代码
# import torch.onnx.symbolic_opset11
print("对应opset文件夹代码在这里：", os.path.dirname(torch.onnx.__file__))

model1 = Model1()
model2 = Model2()
dummy = torch.zeros(1, 1, 3, 3)
torch.onnx.export(model1,  (dummy,), "D:/0/demo1.onnx")
torch.onnx.export(model2,  (dummy,), "D:/0/demo2.onnx")

print("Done.!")