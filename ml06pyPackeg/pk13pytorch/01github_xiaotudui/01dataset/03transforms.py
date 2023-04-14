from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision
from torchvision import transforms

class MyData(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.label_path = label_dir
        self.image_path = image_dir
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img = Image.open(os.path.join(self.image_path, img_name))
        with open(os.path.join(self.label_path, label_name), 'r') as f:
            label = f.readline()
        if self.transform:
            img = transform(img)
        return img, label

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)


transform = transforms.Compose([transforms.Resize(400),
                                #transforms.ToTensor()
                                ])
root_dir = r"D:\02dataset\hymenoptera_data\train/"
image_ants = root_dir + "ants_image"
label_ants = root_dir + "ants_label"
ants_dataset = MyData(image_ants, label_ants, transform=transform)


train_set = torchvision.datasets.CIFAR10(root=r"D:\02dataset\cifar-10-python/", train=True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10(root=r"D:\02dataset\cifar-10-python/", train=False, transform=transform, download=True)

img, target = test_set[0]
img.show()

print("Done")