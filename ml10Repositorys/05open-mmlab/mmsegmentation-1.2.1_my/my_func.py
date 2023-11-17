import cv2
import numpy as np


def save_tensor(my_tensor, img_name="img"):
    tensor = my_tensor.data.cpu().numpy()  # Convert tensor to numpy array
    tensor = np.transpose(tensor, (1, 2, 0))
    if len(tensor.shape) == 3 and tensor.shape[0] < tensor.shape[1] and tensor.shape[0] < tensor.shape[2]:
        tensor = tensor * 255 / np.max(tensor)  # Scale values to 255
    cv2.imwrite(img_name + '.jpg', tensor)





