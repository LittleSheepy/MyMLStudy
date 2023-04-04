import matplotlib.pyplot as plt
import numpy as np


patch_shape = [100,100,3]
diameter = np.minimum(patch_shape[0], patch_shape[1])

x = np.linspace(-1, 1, diameter)
y = np.linspace(-1, 1, diameter)
x_grid, y_grid = np.meshgrid(x, y, sparse=True)
z_grid =  (x_grid ** 2 + y_grid ** 2)** 20

image_mask = 1 - np.clip(z_grid, -1, 1)

image_mask = np.expand_dims(image_mask, axis=2)
image_mask = np.broadcast_to(image_mask, patch_shape)
print(image_mask.shape)
plt.imshow(image_mask)
plt.show()

