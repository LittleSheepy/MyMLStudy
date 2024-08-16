import numpy as np
import matplotlib.pyplot as plt

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

x = np.linspace(-10, 10, 1000)
y = mish(x)

plt.plot(x, y)
plt.title('Mish Activation Function')
plt.xlabel('x')
plt.ylabel('Mish(x)')
plt.grid(True)
plt.show()
