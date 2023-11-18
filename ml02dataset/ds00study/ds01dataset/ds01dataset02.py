import torchvision

# 创建transform，将 PIL Image 转换为 tensor
data_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 建议download始终为True，即使你自己提前下载好了数据集
# 下载慢的话可以拷贝控制台输出的下载地址，然后到迅雷下载好后再将压缩包拷贝至root下即可
# 下载地址也可以在torchvision.datasets.CIFAR10类的源码中查看
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True,
                                         transform=data_trans, download=True)

img, target = train_set[0]
print(type(img))  # <class 'torch.Tensor'>
print(target)     # 6
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(train_set.classes)
print(train_set.classes[target])  # frog