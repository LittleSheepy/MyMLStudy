import matplotlib.pyplot as plt
import numpy as np


a1 = np.arange(10)*10
a2 = np.arange(10)
index = np.argsort(-a1)
print(a1)
print(index)
print(a1[index])
print(a1[index][:5])
print(a1[index][:5][:2])
print("*"*20)

# beach 切片
beach = 2
a = np.arange(4)
np.random.shuffle(a)
a = a.reshape([2,-1]) * 10

print(a)
index = np.argsort(-a)
print(index)

row = np.arange(beach).reshape([beach,1])
dot = a[row,index[:,:3]]

print(dot)

print("*"*20)

index = np.where(a > 40)
ai = a[index]
print(ai)




