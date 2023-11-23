import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
print('')
def use_torch(inputs, target):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(inputs, target)
    print(loss)

def use_none(inputs, target):
    # softmax
    temp1 = torch.exp(inputs) / torch.sum(torch.exp(inputs), dim=1, keepdim=True)
    # log
    temp2 = torch.log(temp1)
    # nll
    temp3 = torch.gather(temp2, dim=1, index=target.view(-1, 1))
    temp4 = -temp3
    output = torch.mean(temp4)
    print(output)

if __name__ == '__main__':
    inputs = torch.tensor([[0.1, 0.2, 0.7]])
    target = torch.tensor([1])
    use_torch(inputs, target)
    use_none(inputs, target)







