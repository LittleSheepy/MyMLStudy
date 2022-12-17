import torch
import torch.nn as nn

class CONV_BN_RELU(nn.Module):
    def __init__(self,in_channel,ou_channel,kernel_size,stride,padding,groups = 1):
        super(CONV_BN_RELU, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel,ou_channel,kernel_size,stride,padding = padding,groups=groups),
                                  nn.BatchNorm2d(ou_channel),
                                  nn.ReLU())
    def forward(self,x):
        return self.conv(x)

class BLOCK(nn.Module):
    def __init__(self,inchannel,out_channel,stride):
        super(BLOCK, self).__init__()
        self.conv = nn.Sequential(
            CONV_BN_RELU(inchannel,inchannel,kernel_size=3,stride = stride,padding = 3//2,groups = inchannel),
            CONV_BN_RELU(inchannel, out_channel, kernel_size=1, stride=1,padding = 0)
        )

    def forward(self,x):
        return self.conv(x)



class Mobilenet_v1(nn.Module):
    def __init__(self,in_channels,classes,alpha):
        super(Mobilenet_v1, self).__init__()
        channels = [32,64,128,256,512,1024]
        channels = [int(i*alpha) for i in channels]
        self.conv1 = CONV_BN_RELU(in_channels,channels[0],3,2,padding = 3//2)

        self.block = BLOCK
        self.stage = nn.Sequential(
            self.block(channels[0], channels[1], 1),
            self.block(channels[1], channels[2], 2),
            self.block(channels[2], channels[2], 1),
            self.block(channels[2], channels[3], 2),
            self.block(channels[3], channels[3], 1),
            self.block(channels[3], channels[4], 2),
            self._make_stage(5,channels[4],channels[4],stride=1),
            self.block(channels[4], channels[5], 2),
            self.block(channels[5], channels[5], 1),
        )
        self.pool = nn.AvgPool2d(7)
        self.fc= nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(channels[-1],classes),
            nn.Softmax()
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode= 'fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)



    def _make_stage(self,num_stage,inchannel,ouchannels,stride):
        strides = [stride]+[1]*(num_stage-1)
        layer = []
        for i in range(num_stage):
           layer.append(self.block(inchannel,ouchannels,strides[i]))
        return nn.Sequential(*layer)

    def forward(self,x):
        x = self.conv1(x)
        x = self.stage(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    input = torch.empty(1,3,224,224)
    m = Mobilenet_v1(3,10,0.5)
    out = m(input)
    print(out)




