import matplotlib.pyplot as plt
import numpy as np


a1 = np.arange(2*2*3)
print(a1)
a2 = np.reshape(a1, (2,2,3))
print(a2)
a3 = np.transpose(a2, (1,2,0))
print(a3)
a4 = np.transpose(a2, (0,2,1))
print(a4)
a5 = np.transpose(a2, (1,0,2))
print(a5)
print("**"*20)
a6 = np.transpose(a2, (2,0,1))
print(a2)
print(a6)

a7 = np.transpose(np.reshape(np.array([np.arange(7)] * 7 * 2),(2, 7, 7)), (1, 2, 0))

a8 = np.reshape(a7, (1,7,7,2))

a9 = np.transpose(a8, (0, 2, 1, 3))
print(a9)