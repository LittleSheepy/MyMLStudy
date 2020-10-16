import sklearn as sk
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

x=[[1],[4],[5],[7],[8]]
y=[2.1,5.1,5.96,7.78,9.2]

print("---LinearRegression---")
"""
def __init__(self, fit_intercept=True,   # 是否有截据，如果没有则直线过原点;
                normalize=False,    # 是否将数据归一化;
                copy_X=True,
                n_jobs=None):
"""
LR = linear_model.LinearRegression(fit_intercept=True)
LR.fit(x, y)
w = LR.coef_
b = LR.intercept_
print(f"w,b = {w},{b}")
x0 = np.arange(0, 10, 1)
#y0 = w * x0 + b
y0 = LR.predict(x0.reshape([-1,1]))
plt.scatter(x, y)
plt.plot(x0, y0)
plt.show()

print("---Ridge---")
"""
min ||XW - y||**2 + alpha * ||W||**2

def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,copy_X=True, 
             max_iter=None,   # 最大迭代次数
             tol=1e-3, 
             solver="auto",   # 求解器，有auto, svd, cholesky, sparse_cg, lsqr几种
             random_state=None):
"""
Ridge_L2 = linear_model.Ridge(fit_intercept=True)
Ridge_L2.fit(x, y)
w = Ridge_L2.coef_
b = Ridge_L2.intercept_
print(f"w,b = {w},{b}")
x0 = np.arange(0, 10, 1)
y0 = w * x0 + b
plt.scatter(x, y)
plt.plot(x0, y0)
plt.show()


















