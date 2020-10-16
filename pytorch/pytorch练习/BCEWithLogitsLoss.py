import torch
nn = torch.nn

#input = torch.randn(3,3)

inputs = torch.FloatTensor([[1.2521,  0.3917,  0.1002],
                           [2.5490, -0.6669,  0.7129],
                           [0.0414,  0.5897, -0.4870]])

print(inputs)
m = nn.Sigmoid()
inputs_sig = m(inputs)
print(input)

target = torch.FloatTensor([[0,1,1],[0,0,1],[1,0,1]])

loss = nn.BCELoss()
l = loss(inputs_sig, target)
print("l",l)

lossl = nn.BCEWithLogitsLoss()

ll = lossl(inputs, target)
print("ll",ll)














