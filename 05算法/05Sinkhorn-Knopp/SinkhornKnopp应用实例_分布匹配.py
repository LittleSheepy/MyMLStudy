# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import pairwise_distances

from Sinkhorn_Knopp import compute_optimal_transport

# two concentric circles
X, y = make_circles(n_samples=100, noise=0.05, factor=0.5, shuffle=False)
X1 = X[y==0]
X2 = X[y==1]

n, m = len(X1), len(X2)


# In[50]:

# SinkhornKnopp应用实例_分布匹配
# Distance metric
M = pairwise_distances(X1, X2, metric='euclidean')
# Uniform weights
n, m = M.shape
r = np.ones(n) / n
c = np.ones(m) / m
P, d = compute_optimal_transport(M, r, c, lam=100, epsilon=1e-6)


# In[51]:


plt.imshow(P)


# In[52]:


fig, ax = plt.subplots()

ax.scatter(X1[:,0], X1[:,1], color='blue', label='set 1')
ax.scatter(X2[:,0], X2[:,1], color='orange', label='set 2')
ax.legend(loc=0)


# In[53]:


fig, ax = plt.subplots()

ax.scatter(X1[:,0], X1[:,1], color='blue', label='set 1')
ax.scatter(X2[:,0], X2[:,1], color='orange', label='set 2')

ax.legend(loc=0)

for i in range(n):
    for j in range(m):
        if P[i,j] > 1e-5:
            ax.plot([X1[i,0], X2[j,0]], [X1[i,1], X2[j,1]], alpha=P[i,j] * n, color='red')


# In[54]:


"""
Interpolate between the two distributions.

Input:
    - alpha : value between 0 and 1 for the interpolation

Output:
    - X : the interpolation between X1 and X2
    - w : weights of the points
"""
alpha = 0.6
mixing = P.copy()
# Normalize, so each row sums to 1 (i.e. probability)
mixing /= r.reshape((-1, 1))
X = (1 - alpha) * X1 + alpha * mixing @ X2
w = (1 - alpha) * r + alpha * mixing @ c

fig, ax = plt.subplots()

ax.scatter(X1[:,0], X1[:,1], color='blue', label='set 1')
ax.scatter(X2[:,0], X2[:,1], color='orange', label='set 2')

ax.scatter(X[:,0], X[:,1], color='red', label='interpolation')
ax.legend(loc=1)

plt.show()
# ## Domain transfer
#
# We make a small classification problem and divide in train and test. The test set is perturbated (shift + noise)

# In[55]:

