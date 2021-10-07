from numpy.core.numeric import cross
import torch.nn as nn
import torch
# loss = nn.CrossEntropyLoss()
# data = torch.randn(5 * 3 * 3, 8)
# target = torch.randint(0, 5, (5 * 3*3,))
# data = data.view(-1, 8)
# target = target.view(-1, 1)
# target = torch.flatten(target)
# print (target.shape)
# output = loss(data, target)
# print(output)


# w1 = torch.FloatTensor([[2.], [1.]])
# w2 = torch.FloatTensor([3.])
# w1.requires_grad = True
# w2.requires_grad = True

# d = torch.matmul(x, w1)
# # d[:] = 1   # 稍微调换一下位置, 就没有问题了

# f = torch.matmul(d, w2.T)
# f.backward()
# print(w2.grad)


import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter

# x = torch.tensor(3.0, requires_grad=True)
# optimizer = optim.SGD([x], lr=0.1)
# y = torch.tensor(4.0)
# z = torch.pow(x, 2) * y
# optimizer.zero_grad()
# z.backward()
# print(x.grad)
# optimizer.step()
# print(x)


# x = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]], requires_grad=True)
# y = x**2 + x

# gradient1 = torch.tensor([[1.,1.,1.],[1.,1.,1.]])
# gradient2 = torch.tensor([[1.,0.1,0.01],[1.,1.,1.]])
# y.backward(gradient1)
# print(x.grad)

# x.grad.zero_()
# y = x**2 + x
# y.backward(gradient2)
# print(x.grad)

# from torch.optim import lr_scheduler
# import torch.optim as optim
# x = torch.tensor(3.0, requires_grad=True)
# optimizer = optim.Adam(params=[x], lr=0.05)
# scheduler2 = lr_scheduler.MultiStepLR(optimizer, [30, 60,100], 0.1)
# x2 = list(range(150))
# y2 = []
# for epoch in range(150):
#     scheduler2.step()
#     print(epoch, scheduler2.get_lr()[0])


# mse_loss = torch.nn.MSELoss()
# a =torch.tensor([1,2,3]).float()
# b = torch.tensor([2,4,6]).float()
# print(mse_loss(a, b))

# a = torch.randn(2,2,3)
# b = torch.randn(2,2,3)
# c = [a,b]
# c = torch.cat(c, 0).shape
# print(c)

# a = torch.randn(1, 138, 3,3)
# b = (torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0]), torch.tensor([  3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,
#          17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,
#          31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
#          45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
#          59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
#          73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,
#          87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,
#         101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
#         115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
#         129, 130, 131]))
# print(a[b].shape)


# a = torch.randn(10,10,5)
# b = torch.randint(0, 5, (10, 10))

# a = a.view(-1, 5)
# b = torch.flatten(b)
# corss_loss = torch.nn.CrossEntropyLoss()
# loss = corss_loss(a, b)
# print(loss)
# f = torch.nn.LogSoftmax(-1)
# aa = f(a)
# print(f"LogSoftmax shape {aa.shape}")
# nll_loss = torch.nn.NLLLoss()
# # aa = nll_loss(aa, b)
# # print(aa)


# mask = torch.ones(10,10)
# mask = torch.flatten(mask)
# sel = torch.where(mask > 0.5)
# sel_neg = torch.where(mask <= 0.5)
# mask[sel] = 1
# mask[sel_neg] = 0

# mask_value = torch.einsum("i,ij->ij", mask, aa)
# mask_label = torch.einsum("i,i->i", mask, b).long()
# print(mask_label)
# print(nll_loss(mask_value, mask_label))

# sel_value = aa[sel]
# sel_label = b[sel]
# print(nll_loss(sel_value, sel_label))


# a = torch.randn(1, 10, 3, 3)
# print(a)
# print(a[0, :, 1])

# 求解距离
# a = torch.arange(100)
# b = torch.arange(100, 200)
# print(a[:, None] - a[None, :])

# L = 4
# a = torch.randint(0, 5, ( L,))
# f = nn.Embedding(5, 3)
# # print(a.unsqueeze(1).expand(-1,5))
# b = f(a)
# print(a.shape)
# print(b.shape)
# c=  b.unsqueeze(1).expand(-1,L, -1)
# d = b.unsqueeze(0).expand(L, -1, -1)
# print(b.unsqueeze(1).shape, c.shape)
# print(b.unsqueeze(0).shape, d.shape)
# print(torch.cat([c, d], dim=-1).shape)
# print(c)
# print(d)

# f = torch.nn.Softmax(-1)
# print(f(torch.randn(2, 8)))


# a = torch.randn(10,10,5)
# b = torch.randint(0, 5, (10, 10))

# a = a.view(-1, 5)
# b = torch.flatten(b)
# corss_loss = torch.nn.CrossEntropyLoss()
# loss = corss_loss(a, b)
# print(loss)
# f = torch.nn.LogSoftmax(-1)
# aa = f(a)
# print(f"LogSoftmax shape {aa.shape}")
# nll_loss = torch.nn.NLLLoss(reduce = False)
# aa = nll_loss(aa, b)
# print(aa)


# mask = torch.randn(10,10)
# mask = torch.flatten(mask)
# sel = torch.where(mask > 0.5)
# sel_neg = torch.where(mask <= 0.5)
# mask[sel] = 1
# mask[sel_neg] = 0

# mask_value = mask * aa
# res = torch.mean(mask_value)
# print(mask_value)
# print(torch.mean(mask_value))

# a = torch.randn(5, 5)
# b = torch.randn(5, 5)
# mask = torch.randn(5,5)
# sel = torch.where(mask > 0.5)
# sel_neg = torch.where(mask <= 0.5)
# mask[sel] = 1
# mask[sel_neg] = 0
# mse_loss = torch.nn.MSELoss(reduce=False)
# c = mse_loss(a, b)
# c = mask * c
# print(torch.mean(c))

# a = torch.arange(100).view(-1, 5)
# print(torch.sum(a, -1))

import torch
# x=torch.ones(2,2,requires_grad=True)
# print(x)
# y=x+2
# print(y)
# #如果一个张量不是用户自己创建的，则有grad_fn属性.grad_fn 属性保存着创建了张量的 Function 的引用
# print(y.grad_fn)
# y.backward()
# print(x.grad)

# L = 10
# B = 2
# idx_pdb = torch.arange(L).long()
# print(idx_pdb.expand((B, L)))

# a = torch.randint(0, 10, (2,2, 5)).float()
# print("a is ", a)
# print("mean is ", a.mean(-1))
# pick = a.mean(-1).argmax(1)
# batchs_idx = torch.arange(2).long()
# print(f"pick {pick}")
# print("final is ", a[batchs_idx,pick])

# a = torch.randint(0, 10, (5,)).float()
# print(a)
# print(torch.clamp(a, 2, 5))

# x = torch.FloatTensor(torch.randn((10,10)))
# y = torch.FloatTensor(torch.randn((10,10)))

# tmp = torch.sum(x * y)
# normx = torch.sqrt(torch.sum(x * x)) * torch.sqrt(torch.sum(y * y))
# similarity = torch.cosine_similarity(x, y, dim = 0)
 
# print(torch.sum(similarity), tmp / normx)
# import numpy as np
# print(np.nan < 14)

# a = torch.arange(12).reshape(-1, 3).float()
# print(torch.mean(a, -1))

# a = torch.arange(12).reshape(-1, 3).float()
# mask = a < 5
# print(mask)
# b = a.masked_fill(mask, -1)
# print(b)

# class Mod(nn.Module):
#     def __init__(self):
#         super(Mod, self).__init__()
#         self.fc = nn.Linear(1, 10)
#         # self.fc2 = nn.ModuleList([nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)])
#         self.fc2 = nn.Linear(10, 10)

#     def forward(self, x):
#         x = self.fc(x)
#         for f in self.fc2:
#             x = f(x)
#         return x


# mod = Mod()
# for i in mod.state_dict():
#     print(i)
# torch.save(mod.state_dict(), "final.pt")
# mod.load_state_dict(torch.load("final.pt"))
# print(mod.state_dict())
# data = torch.randn(2, 1)
# print(mod(data))
# for  i in mod.parameters():
#     print(i)

# import numpy as np
# def get_fa_vec(p1, p2, p3):
#     x1 = p2 - p1
#     x2 = p3 - p1
#     x1 = x1 / np.linalg.norm(x1)
#     x2 = x2 / np.linalg.norm(x2)
#     # cos_theta = np.dot(x1, x2)
#     # v_vertical_x1 = x2 - cos_theta * x1
#     v_vertical_x1 = x2
#     cross_v = np.cross(v_vertical_x1, x1)
#     print(x1, v_vertical_x1, cross_v)
#     print(np.dot(cross_v, x1), np.dot(cross_v, x2), np.dot(cross_v, v_vertical_x1))
# p1 = np.array([0,0., 0])
# p2 = np.array([1,2.,0])
# p3 = np.array([2, 1.,2])
# get_fa_vec(p1, p2, p3)

# a = torch.tensor([2,3,4.], requires_grad=True).float()
# b = torch.tensor([7,2,3.], requires_grad=True).float()
# c = a.detach() * 4 + 6 * b
# # c.requires_grad = True
# c = torch.sum(c)
# # mse_loss = torch.nn.MSELoss()
# # loss = mse_loss(c, b)
# # loss.backward()
# c.backward()
# d.backward()
# print(a.grad)

# import torch
# from torch.utils.data import Dataset
# class DataRead(Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#         self.data = torch.arange(100).view(-1, 20)
#         pass
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.data[index]

# d = DataRead()
# dataloader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=True)
# while 1:
#     print("=" * 20)
#     for i, data in enumerate(dataloader):
#         print(data)


# a = torch.tensor([2,3,4.], requires_grad=True).float()
# b = torch.tensor([7,2,3.], requires_grad=True).float()
# c = a.detach()

# d = torch.sum(a * b)
# e = torch.sum(b * c)
# d.backward()
# e.backward()
# print(a.grad)
import torch.utils.checkpoint as checkpoint
import torch
import torch.nn as nn


# conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1)

# def seg1(x):
#     return conv(x)

# print('查看conv里面的梯度，一开始应当全为0或None')
# print(conv.weight.grad)
# print(conv.bias.grad)

# x = torch.ones(1, 1, 1, 1)
# y = seg1(x).mean() - 3
# y.backward()

# print('查看conv里面的梯度，现在应该不为0了')
# print(conv.weight.grad)
# print(conv.bias.grad)

# print('清空conv的梯度，进行下一次测试')
# conv.weight.grad.data.zero_()
# conv.bias.grad.data.zero_()

# print('查看conv里面的梯度，现在应该全为0了')
# print(conv.weight.grad)
# print(conv.bias.grad)

# y = checkpoint(seg1, x).mean() - 3
# try:
#     print('此时应当会失败，y并不是计算图的一部分，因为x的requires_grad为False，checkpoint认为这段函数是不需要计算梯度的')
#     y.backward()
# except RuntimeError as e:
#     print('backward果然抛出异常了')

# print('查看conv里面的梯度，现在应该保持不变，仍然全为0了')
# print(conv.weight.grad)
# print(conv.bias.grad)

# print('让输入的requires_grad为True，有俩个办法，一个是直接设定x的requires_grad为True，另外一个办法就是与另外一个requires_grad为True的常量合并操作')
# print('这里使用的是合并操作，因为有时候并不能直接设置输入的requires_grad=True，另外我认为合并操作占用的显存更少，因为grad的shape跟原始变量是一样的'
#       '，使用合并操作，额外无用的grad的size只有1，而设定输入的requires_grad为True，额外无用的grad的size跟输入一样大')
# x2 = x + torch.zeros(1, dtype=x.dtype, device=x.device, requires_grad=True)
# y = checkpoint(seg1, x2).mean() - 3
# y.backward()
# print('现在backward不会报错了')
# print('查看conv里面的梯度，现在不为0了')
# print(conv.weight.grad)
# print(conv.bias.grad)
# print('实验完成')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(1000, 1000)
    def forward(self, x, y):
        x = self.fc(x)
        y = self.fc(y)
        return x + y

data = torch.randn(10, 1000)
datay = torch.randn(10, 1000, requires_grad=False)
model = Model()
x = model(data, datay)
# x = checkpoint.checkpoint(model, data, datay)
print(x.requires_grad)