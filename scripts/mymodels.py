'''
mymodels.py published by Maryam on Jun 2021 includes networks used for training data
'''
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Fully connected neural network
class basicRNN(nn.Module):
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


# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu', dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # x: (batch_size, seq, input_size), h0: (num_layers, batch_size, 128)

        # Forward propagate RNN
        # out: tensor of shape (batch_size, seq_length, hidden_size) containing the output features (h_t) from the last layer of the RNN, for each t
        out, _ = self.rnn(x, h0)
        # out: (batch_size, seq, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (batch_size, hidden_size)

        out = self.fc(out)
        # out: (batch_size, num_classes)
        return out


# GRU neural network
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # x: (batch_size, seq, input_size), h0: (num_layers, batch_size, 128)

        # Forward propagate RNN
        # out: tensor of shape (batch_size, seq_length, hidden_size) containing the output features (h_t) from the last layer of the RNN, for each t
        out, _ = self.gru(x, h0)
        # out: (batch_size, seq, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (batch_size, hidden_size)

        out = self.fc(out)
        # out: (batch_size, num_classes)
        return out


# lstm neural network
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # x: (batch_size, seq, input_size), h0: (num_layers, batch_size, 128)

        # Forward propagate RNN
        # out: tensor of shape (batch_size, seq_length, hidden_size) containing the output features (h_t) from the last layer of the RNN, for each t
        out, _ = self.lstm(x, (h0, c0))
        # out: (batch_size, seq, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (batch_size, hidden_size)

        out = self.fc(out)
        # out: (batch_size, num_classes)
        return out


