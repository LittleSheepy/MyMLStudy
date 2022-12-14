import torchvision
from torch import nn
from torch.utils.data import DataLoader
import cv2
from PIL import Image
train_data = torchvision.datasets.CIFAR10("./data",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True
                                          )
trainloader = DataLoader(train_data,
                         batch_size=2,
                         shuffle=True,
                         num_workers=0
                         )
for imgs, lables in trainloader:
    im0 = imgs[0]
    im0 = im0.cpu().clone()
    im0 = torchvision.transforms.ToPILImage()(im0)
    # im0 = Image.fromarray(im0)
    im0.show()
    print(lables)