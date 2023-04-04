import matplotlib.pyplot as plt
import numpy as np

x=np.array([1,2,3,5,6,7,8,9])
print(np.clip(x,3,8))



a = np.arange(16).reshape([4,4])
print(a)
print(np.clip(a,1,[2, 6, 2, 5]))




