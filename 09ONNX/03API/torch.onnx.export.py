import torch
import torch.nn as nn
import os


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, stride=1, padding=1,bias=True)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

# 这个包对应opset11的导出代码，如果想修改导出的细节，可以在这里修改代码
# import torch.onnx.symbolic_opset11
print("对应opset文件夹代码在这里：", os.path.dirname(torch.onnx.__file__))

model = Model()
input = torch.zeros(1, 1, 3, 3)
torch.onnx.export(model,        # 要转换的模型
                    (input,),     # 模型的任意一组输入
                    "demo.onnx",  # 导出的onnx文件
                    opset_version=11,   # ONNX 算子集的版本
                    input_names=['input'],  # 输入tensor的名称
                    output_names=['output'],# 输出tensor的名称
                    )

print("Done.!")