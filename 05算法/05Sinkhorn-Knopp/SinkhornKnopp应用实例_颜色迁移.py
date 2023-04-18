#!/usr/bin/env python
# coding: utf-8
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.neighbors import KNeighborsRegressor

from SinkhornKnopp应用实例_分布匹配 import run_optimal_transport


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


if __name__ == '__main__':
    # get arguments
    n_pixels = 1000
    n_neighbors = 10

    # read the images
    image_from = io.imread(join_path('Figures/PB.jpg')) / 256
    image_to = io.imread(join_path('Figures/PC.jpg')) / 256

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
    fig.show()
    # fig.savefig(join_path('Figures/color_distributions.png'))

    P, d, r, c = run_optimal_transport(X_to_ss, X_from_ss, 10)

    # model transfer
    transfer_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    transfer_model.fit(X_to_ss, np.matmul(P / r.reshape((-1, 1)), X_from_ss))
    X_transfered = transfer_model.predict(X_to)

    image_transferd = minmax(mat2im(X_transfered, shape_to))
    plt.imshow(image_transferd)
    plt.show()
    # io.imsave(join_path('Figures/PB2PC.jpg'), image_transferd)

