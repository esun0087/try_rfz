import torch
import torch.nn as nn
logs = nn.LogSoftmax(dim=1)
nll_loss = criterion = nn.NLLLoss()
c_loss = nn.CrossEntropyLoss()

data= torch.randn(100, 3, 8, 5)
target = torch.randint(0, 5, size = (100, 3,8))
mask = torch.randint(0, 2, size = (100, 3, 8))
sel_mask = torch.where(mask > 0)
sel_data = data[sel_mask]
sel_target = target[sel_mask]
print(sel_data.shape)
print(sel_target.shape)
print(c_loss(data.view(-1, 5), torch.flatten(target)), nll_loss(logs(sel_data), sel_target))
