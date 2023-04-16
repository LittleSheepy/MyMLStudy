import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        return x, y

    def __len__(self):
        return len(self.data)

# 数据
data = [(1, 2), (3, 4), (5, 6)]

# Then you can create an instance of the dataset:
my_dataset = MyDataset(data)

# You can then use a DataLoader to load the data in batches:
my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=2, shuffle=True)

# Now you can iterate over the dataloader to get batches of data:
for batch in my_dataloader:
    x, y = batch
    print(x, y)