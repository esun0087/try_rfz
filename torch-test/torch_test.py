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

a = torch.randint(0, 10, (2,2, 5)).float()
print("a is ", a)
print("mean is ", a.mean(-1))
pick = a.mean(-1).argmax(1)
batchs_idx = torch.arange(2).long()
print(f"pick {pick}")
print("final is ", a[batchs_idx,pick])

# a = torch.randint(0, 10, (5,)).float()
# print(a)
# print(torch.clamp(a, 2, 5))