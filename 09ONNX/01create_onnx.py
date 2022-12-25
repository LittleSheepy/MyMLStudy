import torch
import torch.nn as nn
class network(nn.Module):
    def __init__(self):
        super().__init__()

    def create(self):
        return nn.Sequential(nn.Conv2d(1,1,1),nn.ReLU())

if __name__ == '__main__':
    net = network()
    model = net.create()
    input = torch.randn([1,1,1,1])
    torch.onnx.export(model,args=input,f="./torch.onnx", opset_version=11)