import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

# 定义数据预处理和增强的transform
transform = transforms.Compose([
    transforms.Resize((256, 512)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor格式
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 图像标准化
])
tar_transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor格式
])
# 加载Cityscapes数据集
dataset = Cityscapes(root=r'F:\sheepy\00MyMLStudy\ml10Repositorys\05open-mmlab\mmsegmentation-1.2.1_my\data\cityscapes', split='train', mode='fine', target_type='semantic', transform=transform, target_transform=tar_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# 定义模型和损失函数
model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
criterion = torch.nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 训练模型
for epoch in range(10):
    for batch in dataloader:
        images, labels = batch['image'], batch['label']

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs['out'], labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练进度
        print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")