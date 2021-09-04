import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
train_x = [torch.tensor([1, 2, 3, 4, 5, 6, 7]),
           torch.tensor([2, 3, 4, 5, 6, 7]),
           torch.tensor([3, 4, 5, 6, 7]),
           torch.tensor([4, 5, 6, 7]),
           torch.tensor([5, 6, 7]),
           torch.tensor([6, 7]),
           torch.tensor([7])]

class MyData(Dataset):
    def __init__(self, train_x):
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, item):
        return self.train_x[item]
net = nn.LSTM(1, 5, batch_first=True)
def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data.unsqueeze(-1).float(), data_length  # 对train_data增加了一维数据
train_data = MyData(train_x)
train_dataloader = DataLoader(train_data, batch_size=2, collate_fn=collate_fn)

for data, length in train_dataloader:
    # data = rnn_utils.pack_padded_sequence(data, length, batch_first=True)
    output, hidden = net(data)
    # output, out_len = rnn_utils.pad_packed_sequence(output, batch_first=True)
    print("=" * 100)
    print(output.shape)
    break