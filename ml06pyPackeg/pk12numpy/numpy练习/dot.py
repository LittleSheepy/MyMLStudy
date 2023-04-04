import numpy as np

l0 = [[2],[2],[2]]
l00 = [[2,2,2]]
l000 = [[[2,2,2]]]
l1 = [2,2,2]
l2 = [[1,3],[1,3],[1,3]]
l0 = np.array(l0)
l00 = np.array(l00)
l000 = np.array(l000)
l1 = np.array(l1)
l2 = np.array(l2)

d = np.dot(l1,l2)
e = np.dot(l00,l2)
f = np.dot(l000,l2)
print(l0.shape)
print(l00.shape)
print(l000.shape)
print(l1.shape)
print(l2.shape)
print("------------------")
print(d)
print(e)
print(f)
print(d+e)
print(d+f)

