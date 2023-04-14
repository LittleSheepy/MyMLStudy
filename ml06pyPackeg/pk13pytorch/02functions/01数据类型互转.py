import torchvision
from torch import nn
from torch.utils.data import DataLoader
import cv2
from PIL import Image
from torchvision import transforms
import torch

def PIL2Tensor(imgPIL):
    image = transforms.ToTensor()(imgPIL)
    image = image.to("cuda", torch.float)
    return image

def Tensor2PIL(imgTensor):
    image = imgTensor.cpu().clone()
    image = transforms.ToPILImage()(image)
    return image



if __name__ == '__main__':
    imgPath = r"D:\02dataset\testImage\person.jpg"
    img_pil = Image.open(imgPath)
    img_tensor = PIL2Tensor(img_pil)
    img_pil_new = Tensor2PIL(img_tensor)
    img_pil_new.show()
