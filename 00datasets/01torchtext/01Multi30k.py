import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
"""We create the tokenizers."""
def tokenize(text):
    # Tokenizes German text from a string into a list of strings
    return text.split(" ")

SRC = Field(tokenize = tokenize,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)
"""Load the data."""
dataRoot = r"F:\01dataset\00publicDataSet/"
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),fields = (SRC, SRC), root=dataRoot)

"""Build the vocabulary."""
SRC.build_vocab(train_data, min_freq = 2)
# SRC.build_vocab(train_data, min_freq = 1)# 37136
INPUT_DIM = len(SRC.vocab)
print("个数：",INPUT_DIM)

"""Define the device."""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Create the iterators."""
BATCH_SIZE = 10
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

for i, batch in enumerate(train_iterator):
    src = batch.src     # torch.Size([17, 10])
    trg = batch.trg     # trg = [trg_len, batch_size]
    print(src)
    break
