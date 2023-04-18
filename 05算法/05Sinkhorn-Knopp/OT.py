#!/usr/bin/env python
# coding: utf-8
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import os
from sklearn.datasets import make_circles

from Sinkhorn_Knopp import compute_optimal_transport



from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


Xtr, Xte, ytr, yte = train_test_split(*make_blobs(n_samples=200, centers=3, cluster_std=1.5), test_size=0.5)

Xte += np.random.randn(*Xte.shape) * 2 + np.array([[2, -3]]) 


# In[56]:


fig, ax = plt.subplots()

for cl, col in zip(range(3), ['blue','orange', 'green']):
    X = Xtr[ytr==cl,:]
    ax.scatter(X[:,0], X[:,1], color=col, label='class {}'.format(cl+1))
    
ax.scatter(Xte[:,0], Xte[:,1], color='gray', alpha=0.8, label='test') 
ax.legend(loc=0)


# In[57]:


# Distance metric
M = pairwise_distances(Xtr, Xte, metric='euclidean')
# Uniform weights
n, m = M.shape
r = np.ones(n) / n
c = np.ones(m) / m
P, d = compute_optimal_transport(M, r, c, lam=50, epsilon=1e-6)


# In[58]:


fig, ax = plt.subplots()

for i in range(Xtr.shape[0]):
    for j in range(Xte.shape[0]):
        if P[i,j] > 1e-4:
            ax.plot([Xtr[i,0], Xte[j,0]], [Xtr[i,1], Xte[j,1]], color='red', alpha=P[i,j]*100)
            
for cl, col in zip(range(3), ['blue','orange', 'green']):
    X = Xtr[ytr==cl,:]
    ax.scatter(X[:,0], X[:,1], color=col, label='class {}'.format(cl+1))
    
ax.scatter(Xte[:,0], Xte[:,1], color='gray', alpha=0.8, label='test')
            
ax.legend(loc=0)


# ## Color transfer
# 
# See `color_transfer.py` module!

# Image from:
# 
# ![Princess Caroline](Figures/PC.jpg)
# 
# Image to:
# 
# ![Mr. Peanutbutter](Figures/PB.jpg)

# In[59]:


import numpy as np
from skimage import io
from skimage.color import rgb2hsv, hsv2rgb
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import warnings
warnings.simplefilter("ignore", UserWarning)

# get arguments
name_from = 'Figures/PB.jpg'
name_to = 'Figures/PC.jpg'
name_out = 'Figures/PB2PC.jpg'
n_pixels = 1000
lam = 10
n_neighbors = 10
distance_metric = 'euclidean'

def im2mat(I):
    """Converts and image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))

def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)

def minmax(I):
    return np.clip(I, 0, 1)

def join_path(path):
    return os.path.join(os.path.abspath('.'), path)


# In[60]:


# read the images
image_from = io.imread(join_path(name_from)) / 256
image_to = io.imread(join_path(name_to)) / 256

# get shapes
shape_from = image_from.shape
shape_to = image_to.shape

# flatten
X_from = im2mat(image_from)
X_to = im2mat(image_to)

# subsample, only retain n_pixels pixels.
rng = np.random.default_rng(0)
X_from_ss = rng.choice(X_from, size=n_pixels, axis=0, replace=False, shuffle=False)
X_to_ss = rng.choice(X_to, size=n_pixels, axis=0, replace=False, shuffle=False)

fig, axes = plt.subplots(nrows=2, figsize=(5, 10))
for ax, X in zip(axes, [X_from_ss, X_to_ss]):
    ax.scatter(X[:,0], X[:,1], color=X)
    ax.set_xlabel('red')
    ax.set_ylabel('green')
axes[0].set_title('distr. from')
axes[1].set_title('distr. to')
fig.tight_layout()
fig.savefig(join_path('Figures/color_distributions.png'))


# In[61]:


# optimal tranportation
# Distance metric: 'euclidean'
M = pairwise_distances(X_to_ss, X_from_ss, metric=distance_metric)
# Uniform weights
n, m = M.shape
r = np.ones(n) / n
c = np.ones(m) / m
P, d = compute_optimal_transport(M, r, c, lam=lam, epsilon=1e-6)


# In[62]:


# model transfer
transfer_model = KNeighborsRegressor(n_neighbors=n_neighbors)
transfer_model.fit(X_to_ss, np.matmul(P / r.reshape((-1, 1)), X_from_ss))
X_transfered = transfer_model.predict(X_to)

image_transferd = minmax(mat2im(X_transfered, shape_to))
io.imsave(join_path(name_out), image_transferd)


# Result:
# 
# ![Colors of mr. Peanutbutter to Princess caroline.](Figures/PB2PC.jpg)

# ## References
# 
# Lévy, B. and Schwindt, E. (2017). *Notions of optimal transport theory and how to implement them on a computer* [arxiv](https://arxiv.org/pdf/1710.02634.pdf)
# 
# Courty, N., Flamary, R., Tuia, D. and Rakotomamonjy, A. (2016). *Optimal transport for domain adaptation*
# 
# Cuturi, M. (2013) *Sinkhorn distances: lightspeed computation of optimal transportation distances*
# 
