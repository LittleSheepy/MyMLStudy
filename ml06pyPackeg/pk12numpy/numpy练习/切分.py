import matplotlib.pyplot as plt
import numpy as np

a = [[11,12, 13, 14, 15, 16],
     [21,22, 23, 24, 25, 26]]
array = np.array(a)
# print(array[[0,1],[0,1]])
# print(array[:,[0,1]])
# print(array[:,0::2])
# print(array[:,1::2])


a = [[[1, 2, 3], [14, 15, 16]],
     [[21,22, 23], [34, 35, 36]]]
print(np.shape(a))
array = np.array(a)
print("array[0,:, 0]          = ", array[0,:, 0])
print("array[[0],:, 0]        = ", array[[0],:, 0])
print("array[[[0]],:, 0]      = ", array[[[0]],:, 0])
print("array[[[0]],:, [0]]    = ", array[[[0]],:, [0]])
print("array[[[0]],:, [[0]]]  = ", array[[[0]],:, [[0]]])