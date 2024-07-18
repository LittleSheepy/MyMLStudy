import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max
    # pred_dist torch.Size([40, 16])
    # target torch.Size([10, 4]) target.view(-1):torch.Size([40])
    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss 分布焦点损失(DFL)在广义焦点损失中提出
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)      # 5.6407
        tl = target.long()  # target left   5
        tr = tl + 1  # target right         6
        wl = tr - target  # weight left     0.3593
        wr = 1 - wl  # weight right         0.6407
        loss_left_ce = F.cross_entropy(pred_dist, tl.view(-1), reduction="none")    # 1.2893
        loss_left_ce_reshape = loss_left_ce.view(tl.shape)
        loss_left_weight = loss_left_ce_reshape * wl                                # 0.4633

        loss_right_ce = F.cross_entropy(pred_dist, tr.view(-1), reduction="none")
        loss_right_ce_reshape = loss_right_ce.view(tr.shape)
        loss_right_weight = loss_right_ce_reshape * wr
        loss_none = loss_left_weight + loss_right_weight
        loss_mean = (loss_none).mean(-1, keepdim=True)
        return loss_mean

def DFLossTest():
    df_loss = DFLoss()
    inputs = np.array([[-0.9746, -0.7520, 1.4404, 2.8555, 2.5996, 8.3359, 9.2969, 3.2305, 1.1816, 0.2476, -0.5889, -1.2119, -1.4170,
            -2.4102, -2.8711, -2.9629]])
    inputs = torch.from_numpy(inputs)
    targets = np.array([5.6407])
    targets = torch.from_numpy(targets)
    loss = df_loss(inputs, targets)
    pass

if __name__ == '__main__':
    DFLossTest()