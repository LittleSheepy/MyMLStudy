import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class test_net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.a_Linear = nn.Linear(1, 2)
        self.a_Linear.weight.data = torch.Tensor([[5.], [7.]])
        self.a_Linear.bias.data = torch.Tensor([3.])
        self.a_Linear.weight.requires_grad = True
        self.a_Linear.bias.requires_grad = True

        self.b_Linear = nn.Linear(2, 2)
        self.b_Linear.weight.data = torch.Tensor([[5., 6.], [7., 8.]])
        self.b_Linear.bias.data = torch.Tensor([9., 10.])
        self.b_Linear.weight.requires_grad = True
        self.b_Linear.bias.requires_grad = True

        self.c_Linear = nn.Linear(2, 1)
        self.c_Linear.weight.data = torch.Tensor([[1., 2.]])
        self.c_Linear.bias.data = torch.Tensor([3.])
        self.c_Linear.weight.requires_grad = True
        self.c_Linear.bias.requires_grad = True

        self.d_Linear = nn.Linear(2, 1)
        self.d_Linear.weight.data = torch.Tensor([[7., 8.]])
        self.d_Linear.bias.data = torch.Tensor([9.])
        self.d_Linear.weight.requires_grad = True
        self.d_Linear.bias.requires_grad = True

    def forward(self, x):
        x = self.a_Linear(x)
        x = self.b_Linear(x)
        x1 = self.d_Linear(x)
        x2 = self.c_Linear(x)
        return x1 + x2


print("")
set_random_seed(1)
model = test_net()
# print([n for n,p in model.named_parameters()])
optimizer = torch.optim.SGD(
    params=[{'params': [p for n, p in model.named_parameters() if 'a_Linear' in n or 'd_Linear' in n]}], lr=0.1)
optimizer.zero_grad()
a = torch.tensor([2.], requires_grad=False)
loss = model(a)
for n, p in model.named_parameters():
    print(f"{n}'s weight: {p.data}")
    print(f"{n}'s grad: {p.grad}")
print('done.')
loss.backward()
optimizer.step()
print("==================================================")
for n, p in model.named_parameters():
    print(f"{n}'s weight: {p.data}")
    print(f"{n}'s grad: {p.grad}")
print('done.')

"""
================================================
a_Linear.weight's weight: tensor([[5.], [7.]])
a_Linear.weight's grad: None
a_Linear.bias's weight: tensor([3.])
a_Linear.bias's grad: None
b_Linear.weight's weight: tensor([[5., 6.], [7., 8.]])
b_Linear.weight's grad: None
b_Linear.bias's weight: tensor([ 9., 10.])
b_Linear.bias's grad: None
c_Linear.weight's weight: tensor([[1., 2.]])
c_Linear.weight's grad: None
c_Linear.bias's weight: tensor([3.])
c_Linear.bias's grad: None
d_Linear.weight's weight: tensor([[7., 8.]])
d_Linear.weight's grad: None
d_Linear.bias's weight: tensor([9.])
d_Linear.bias's grad: None
done.
"""

"""c_Linear d_Linear false，注册
>>>>>requires_grad=True 注册
a_Linear.weight's weight: tensor([[-17.0000], [-18.6000]])
a_Linear.weight's grad: tensor([[220.], [256.]])
a_Linear.bias's weight: tensor([-20.8000])
a_Linear.bias's grad: tensor([238.])

>>>>> requires_grad=True 没注册
b_Linear.weight's weight: tensor([[5., 6.], [7., 8.]])
b_Linear.weight's grad: tensor([[104., 136.], [130., 170.]])
b_Linear.bias's weight: tensor([ 9., 10.])
b_Linear.bias's grad: tensor([ 8., 10.])

>>>>> requires_grad=False 没注册
c_Linear.weight's weight: tensor([[1., 2.]])
c_Linear.weight's grad: None
c_Linear.bias's weight: tensor([3.])
c_Linear.bias's grad: None

>>>>> requires_grad=False 注册
d_Linear.weight's weight: tensor([[7., 8.]])
d_Linear.weight's grad: None
d_Linear.bias's weight: tensor([9.])
d_Linear.bias's grad: None
done.
"""

"""c_Linear d_Linear True，注册
a_Linear.weight's weight: tensor([[-17.0000], [-18.6000]])
a_Linear.weight's grad: tensor([[220.], [256.]])
a_Linear.bias's weight: tensor([-20.8000])
a_Linear.bias's grad: tensor([238.])

>>>>> requires_grad=True 没注册
b_Linear.weight's weight: tensor([[5., 6.], [7., 8.]])
b_Linear.weight's grad: tensor([[104., 136.], [130., 170.]])
b_Linear.bias's weight: tensor([ 9., 10.])
b_Linear.bias's grad: tensor([ 8., 10.])

>>>>>requires_grad=True 没注册
c_Linear.weight's weight: tensor([[1., 2.]])
c_Linear.weight's grad: tensor([[176., 237.]])
c_Linear.bias's weight: tensor([3.])
c_Linear.bias's grad: tensor([1.])

>>>>>requires_grad=True 注册
d_Linear.weight's weight: tensor([[-10.6000, -15.7000]])
d_Linear.weight's grad: tensor([[176., 237.]])
d_Linear.bias's weight: tensor([8.9000])
d_Linear.bias's grad: tensor([1.])
done.
"""

"""c_Linear d_Linear True，不注册
a_Linear.weight's weight: tensor([[-17.0000], [-18.6000]])
a_Linear.weight's grad: tensor([[220.], [256.]])
a_Linear.bias's weight: tensor([-20.8000])
a_Linear.bias's grad: tensor([238.])

b_Linear.weight's weight: tensor([[5., 6.], [7., 8.]])
b_Linear.weight's grad: tensor([[104., 136.], [130., 170.]])
b_Linear.bias's weight: tensor([ 9., 10.])
b_Linear.bias's grad: tensor([ 8., 10.])

c_Linear.weight's weight: tensor([[1., 2.]])
c_Linear.weight's grad: tensor([[176., 237.]])
c_Linear.bias's weight: tensor([3.])
c_Linear.bias's grad: tensor([1.])

d_Linear.weight's weight: tensor([[7., 8.]])
d_Linear.weight's grad: tensor([[176., 237.]])
d_Linear.bias's weight: tensor([9.])
d_Linear.bias's grad: tensor([1.])
done.
"""