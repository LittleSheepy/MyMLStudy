import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
BCELoss:binary cross entropy 二元交叉熵
"""
def binary_cross_entropy_with_logits_numpy(predictions, targets):
    # 计算sigmoid激活
    sigmoid_predictions = 1 / (1 + np.exp(-predictions))        # [0.52497919 0.549834   0.66818777]
    # 计算二进制交叉熵
    loss_1 = -(targets * np.log(sigmoid_predictions) + (1 - targets) * np.log(1 - sigmoid_predictions))    # [-0.74439666 -0.79813887 -0.40318605]
    loss = np.sum(loss_1)  # 1.9457215783406205
    loss_mean = loss / len(predictions)  # 返回平均损失 0.6485738594468735
    return loss_1

def binary_cross_entropy_with_logits_numpy_test():
    predictions = np.array([0.1, 0.2, 0.7])
    targets = np.array([0, 0, 1])
    loss = binary_cross_entropy_with_logits_numpy(predictions, targets)
    print(loss)

def binary_cross_entropy_with_logits_torch_test():
    pred = torch.tensor([[0.1, 0.2, 0.7]])
    label = torch.tensor([[0.0, 0.0, 1.0]])
    loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")    # reduction="mean"
    print(loss)

def binary_cross_entropy_with_logits_nn_test():
    pred = torch.tensor([[0.1, 0.2, 0.7]])
    label = torch.tensor([[0.0, 0.0, 1.0]])
    loss = nn.BCEWithLogitsLoss(reduction="none")(pred, label)    # reduction="mean"
    print(loss)

# sigmoid + BCELoss = BCEWithLogitsLoss
def binary_cross_entropy_nn_test():
    pred = torch.tensor([[0.1, 0.2, 0.7]])
    label = torch.tensor([[0.0, 0.0, 1.0]])
    pred_sigmoid = torch.sigmoid(pred)
    print(pred_sigmoid)
    loss = nn.BCELoss(reduction="none")(pred_sigmoid, label)    # reduction="mean"
    print(loss)

if __name__ == '__main__':
    binary_cross_entropy_with_logits_numpy_test()
    binary_cross_entropy_with_logits_torch_test()
    binary_cross_entropy_with_logits_nn_test()
    binary_cross_entropy_nn_test()