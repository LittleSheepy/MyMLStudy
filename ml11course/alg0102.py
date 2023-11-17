import numpy as np
import math
import pylab
import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt
n = np.arange(1, 6, 1)
n_factory = [math.factorial(nn) for nn in n] #对应复杂度O(n!)
plt.plot(n, n_factory, label = "n!")
plt.annotate(
        text="O(n!)",
        xy=(4.8, 100), xytext=(-20, 10),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
constant_n_power = 2**n  #对应复杂度O(c^n),c=2
plt.plot(n, constant_n_power, label="c^n")
plt.annotate(
        text="O(c^n, c=2)",
        xy=(4.9, 31.5), xytext=(-15, 10),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
n_with_constant_power = n ** 2 #对应复杂度O(n^c) c = 2
plt.plot(n, n_with_constant_power, label="n^c")
plt.annotate(
        text="O(n^c, c=2)",
        xy=(4.5, 19.7), xytext=(-20, 10),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
n_lg_n = [nn * math.log(nn) for nn in n] #对应复杂度O(n*lg(n))
plt.plot(n, n_lg_n, label="n*lg(n)")
plt.annotate(
        text="O(n*lg(n))",
        xy=(4.3, 6.5), xytext=(-20, 10),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
constant_2 = [2 for nn in n] #对应复杂度O(c), c=2
plt.plot(n, constant_2, label="O(c)")
plt.annotate(
        text="O(c), c=2",
        xy=(4.0, 1.9), xytext=(-20, -20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.legend()
plt.show()