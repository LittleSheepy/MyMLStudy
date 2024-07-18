import torch
import numpy as np
from torch.nn import functional as F

"""
    use numpy
"""
# 定义softmax函数
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# 利用numpy计算
def cross_entropy_np(x, y):
    x_softmax = [softmax(x[i]) for i in range(len(x))]
    x_log = [-np.log(x_softmax[i][y[i]]) for i in range(len(y))]
    loss = np.sum(x_log) / len(y)
    return x_log


# 另外一种实现方式，y转成onehot形式，比较直观
def cross_entropy_np2(x, y):
    # softmax
    num_data, num_class = x.shape
    # log
    log_p = np.array([np.log(softmax(x[i])) for i in range(num_data)])
    y_onehot = np.eye(num_class)[y]
    loss_none = y_onehot * log_p
    loss = - np.sum(loss_none) / num_data
    return loss_none

"""
    use functional
"""
# 调用Pytorch的nn.CrossEntropy计算
def cross_entropy_F(x, y):
    x_pth = torch.from_numpy(x)
    y_pth = torch.from_numpy(y).long()
    loss = F.cross_entropy(x_pth, y_pth, reduction="none")
    return loss

"""
    use nn
"""
def cross_entropy_nn(inputs, target):
    loss = torch.nn.CrossEntropyLoss(reduction="none")(inputs, target)
    return loss

"""
    use torch
"""
def cross_entropy_torch(inputs, target):
    # softmax
    temp1 = torch.exp(inputs) / torch.sum(torch.exp(inputs), dim=1, keepdim=True)   # tensor([[0.2546, 0.2814, 0.4640]])
    # log
    temp2 = torch.log(temp1)    # tensor([[-1.3679, -1.2679, -0.7679]])
    # nll
    temp3 = torch.gather(temp2, dim=1, index=target.view(-1, 1))    # tensor([[-1.2679]])
    temp4 = -temp3
    output = torch.mean(temp4)
    print(output)

if __name__ == '__main__':
    # 假设有数据x, y
    # x = np.array([[0.093, 0.1939, -1.0649, 0.4476, -2.0769],
    #               [-1.8024, 0.3696, 0.7796, -1.0346, 0.473],
    #               [0.5593, -2.5067, -2.1275, 0.5548, -1.6639]])

    # x = np.array([
    #     [0.1, 0.2, 0.7],
    #     [0.1, 0.2, 0.7],
    #     [0.1, 0.2, 0.7],
    # ])
    # y = np.array([0, 0, 1])
    # print('numpy result: ', cross_entropy_np(x, y)) # numpy result:  1.0155949508195155
    # # print('numpy result2: ', cross_entropy_np2(x, y)) # numpy result: 1.0156
    # print('numpy result2: ', cross_entropy_F(x, y)) # numpy result: 1.0156
    x = np.array([[-0.9746, -0.7520, 1.4404, 2.8555, 2.5996, 8.3359, 9.2969, 3.2305, 1.1816, 0.2476, -0.5889, -1.2119, -1.4170,
            -2.4102, -2.8711, -2.9629]])
    y = np.array([5])

    print('numpy result: ', cross_entropy_np2(x, y)) # numpy result:  1.0155949508195155




