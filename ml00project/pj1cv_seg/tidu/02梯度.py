import numpy as np
from matplotlib import pyplot as plt
import pywt
from PIL import Image
def Degrade(image):
    beta =  0.5 * np.random.rand(1)
    image.astype('float32')

    G_col =  np.random.normal(0, beta, image.shape[1])
    G_noise = np.tile(G_col, (image.shape[0],1))
    G_noise = np.reshape(G_noise,image.shape)

    image_G = image + G_noise
    return image_G, beta[0]

image_path = r'D:\04DataSets\04\box_center.jpg'
img = Image.open(image_path)
img = np.array(img)/255.0
img, _ = Degrade(img)
horizental = img[:, :-1] - img[:, 1:]
vertical = img[:-1, :] - img[1:, :]

print(horizental.shape)
plt.subplot(1, 3, 1)
plt.imshow(img, cmap="Greys")
plt.subplot(1, 3, 2)
plt.imshow(horizental, cmap="Greys")
plt.subplot(1, 3, 3)
plt.imshow(vertical, cmap="Greys")
plt.show()
