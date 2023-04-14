import torch
import torch.nn as nn
inp = torch.tensor([[[[1.0, 2, 3, 4, 5, 6],
                      [7, 8, 9, 10, 11, 12],
                      [13, 14, 15, 16, 17, 18],
                      [19, 20, 21, 22, 23, 24],
                      [25, 26, 27, 28, 29, 30],
                      [31, 32, 33, 34, 35, 36],
                      ]]])  # torch.Size([1, 1, 6, 6])
print('inp=')
print(inp)

unfold = nn.Unfold(kernel_size=3, padding=0, stride=3)
inp_unf = unfold(inp)   # torch.Size([1, 9, 4])
print('inp_unf=')
print(inp_unf)
print("Done")

"""
tensor([[[ 1.,  4., 19., 22.],
         [ 2.,  5., 20., 23.],
         [ 3.,  6., 21., 24.],
         [ 7., 10., 25., 28.],
         [ 8., 11., 26., 29.],
         [ 9., 12., 27., 30.],
         [13., 16., 31., 34.],
         [14., 17., 32., 35.],
         [15., 18., 33., 36.]]])
"""
