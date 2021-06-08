'''
import torch
import torch.nn.functional as nn

input_tensor = torch.rand(296, 4, 6)
hidden_tensor = torch.rand(4, 128)

combined = torch.cat((input_tensor[0], hidden_tensor), -1)
print(combined.size())

print(combined)


## nn.Linear
import torch
import torch.nn as nn
m = nn.Linear(20, 30)
print(m)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
torch.Size([128, 30])

'''

import torch
import torch.nn as nn
m = torch.nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()


# input is of size N x C = 3 x 5
input = torch.randn(3, 5, requires_grad=True)
print(input)
print(input.size())
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])
print(target)
print(target.size())
aa = m(input)
output = loss(aa, target)
print(aa.size())
print(output)
output.backward()


