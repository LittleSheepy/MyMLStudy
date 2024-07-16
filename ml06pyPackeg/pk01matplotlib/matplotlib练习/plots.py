import matplotlib.pyplot as plt
import numpy as np


for i in range(6):
    ax = plt.subplot(3,2,i+1)
    ax.set_title(f"ax_{i+1}")
    ax.plot([i for i in range(i+4)])
plt.show()


plt.title("plt")
plt.plot([i for i in range(4)])
plt.show()

plt.title("plt10")
plt.plot([i for i in range(4)],[i+10 for i in range(4)])
plt.show()

plt.title("img")
plt.imshow(np.array([[0,1],[1,0]]),"gray")
plt.show()

plt.title("img")
plt.imshow([[0,1],[1,0]],plt.cm.gray)
plt.show()
