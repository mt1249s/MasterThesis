'''
mymodels.py published by Maryam on Jun 2021 includes networks used for training data
'''
import torch
import torch.nn as nn

# Fully connected neural network
class basicRNN(nn.Module):
    # nn.RNN
    def __init__(self, input_size, hidden_size, num_classes):
        super(basicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, num_classes)
        #self.softmax = nn.LogSoftmax(dim=1)
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        self.relu = nn.ReLU
        output = self.i2o(combined)
        self.relu = nn.ReLU
        #output = self.softmax(output)
        # output = self.sigmoid(output)
        return output, hidden

'''
# Fully connected neural network with one hidden layer
class one_hidden_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(one_hidden_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, num_classes)


        self.l1 = nn.linear(input_size, hidden_size)
        self.relu = nn.ReLU
        self.l2 = nn.linear(hidden_size, num_classes)

    def forward(self, input_tensor, hidden_tensor):
        out = self.l1(input_tensor)
        out = self.relu(out)
        out = self.l2(out)
        
        return out
'''