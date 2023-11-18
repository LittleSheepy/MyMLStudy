import torch
from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(self):
        # 创建5*2的数据集
        self.data = torch.tensor([[1, 2], [3, 4], [2, 1], [3, 4], [4, 5]])
        # 5个数据的标签
        self.label = torch.tensor([0, 1, 0, 1, 2])

    # 根据索引获取data和label
    def __getitem__(self, index):
        return self.data[index], self.label[index]  # 以元组的形式返回

    # 获取数据集的大小
    def __len__(self):
        return len(self.data)


data = myDataset()
# print("")
print("=dataset==================================")
print(f'data size is : {len(data)}')

print(data[1])  # 获取索引为1的data和label

from torch.utils.data import DataLoader

data = myDataset()

print("=DataLoader==================================")
my_loader = DataLoader(data, batch_size=2, shuffle=False, num_workers=0, drop_last=True)
for step, train_data in enumerate(my_loader):
    Data, Label = train_data
    print("step:", step)
    print("data:", Data)
    print("Label:", Label)
