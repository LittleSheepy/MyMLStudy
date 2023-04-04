
import numpy as np
#a = [[1,2]]
a = [[[1],[2]]]
b = [[[11,12],[21,22]]]

a = np.array(a)
b = np.array(b)
print(a.shape, b.shape)

# print(np.hstack([a,b]))
print(np.append(b, a, axis=2))

mon = np.zeros((2000, 100,300,3))
memory = np.zeros((100,50,3))
memory1 = np.zeros((100,50,3))
memory1[0][0][0] = 10
transition = np.hstack((memory,memory1,memory))
print(memory.shape, memory1.shape,transition.shape)