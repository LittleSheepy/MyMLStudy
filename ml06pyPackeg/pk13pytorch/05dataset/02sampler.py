import torch
from torch.utils.data import Dataset, Sampler

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        return x, y

    def __len__(self):
        return len(self.data)

class MySampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

# To use this dataset with a custom sampler, you would first need to create a list of data points:
data = [(1, 2), (3, 4), (5, 6)]

# Then you can create an instance of the dataset:
my_dataset = MyDataset(data)

# You can then create an instance of the sampler:
my_sampler = MySampler(my_dataset)

# You can then use a DataLoader with the custom sampler to load the data in a custom order:
my_dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=2, sampler=my_sampler)

# Now you can iterate over the dataloader to get batches of data in the custom order:
for batch in my_dataloader:
    x, y = batch
    print(x, y)