'''
import torch
import torch.nn.functional as nn

input_tensor = torch.rand(1, 10, 6)
hidden_tensor = torch.rand(1, 128, 6)

combined = torch.cat((input_tensor, hidden_tensor), 1)

print(combined)
'''

## nn.Linear
import torch
import torch.nn as nn
m = nn.Linear(20, 30)
print(m)
input = torch.randn(128, 20)
output = m(input)
print(output.size())
torch.Size([128, 30])


