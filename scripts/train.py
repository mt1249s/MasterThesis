'''
train.py published by Maryam on Jun 2021 includes train & test function
NOte: hyper-parameters of the model should be set in this script, then the model is being called
'''
import torch
import torch.nn as nn
import mymodels


torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# embedding
num_embeddings = 5
embedding_dim = 6

em = torch.nn.Embedding(num_embeddings, embedding_dim).to(device)

import numpy as np
# Hyper-parameters
input_size = embedding_dim
learning_rate = 0.001
batch_size = 125
hidden_size = 100
num_classes = 2
num_layers = 2



# model = mymodels.RNN(input_size, hidden_size, num_layers, num_classes).to(device)
# model = mymodels.GRU(input_size, hidden_size, num_layers, num_classes).to(device)
model = mymodels.LSTM(input_size, hidden_size, num_layers, num_classes).to(device)


criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

batch_mean_dis = []
def train_model(sample, target):
    input_tensor = em(sample)  # input_tensor: (seq, batch_size, input_size)
    input_tensor = torch.reshape(input_tensor, (batch_size, sample.size(0), input_size)).to(device)   # x: (batch_size, seq, input_size),
    prediction = model(input_tensor)
    #print(f'prediction: {prediction}
    #m = nn.Softmax(dim=1)
    #sotfmax_output = m(prediction)
    #print(f'softmax output: {sotfmax_output}')
    loss = criterion(prediction, target)
    guess = torch.argmax(prediction, dim=1)
    #print(f'guess: {guess}, target: {target}')
    init_par = list(model.parameters())
    # print(f'parameters before update: {list(model.parameters())}')
    optimizer.zero_grad()  # Zero the gradients while training the network
    loss.backward()  # compute gradients
    #print('gradient descent after backward')
    #print([p.grad for p in model.parameters()])
    optimizer.step()  # updates the parameters
    #print(f'parameters after update: {list(model.parameters())}')
    updated_par = list(model.parameters())
    # print(len(init_par))
    for i in range(len(init_par)):
        dis = abs(init_par[i] - updated_par[i])
        mean_dis = dis.sum().item()/len(dis)
    temp = mean_dis / len(init_par)
    batch_mean_dis.append(temp)

    return guess, loss.item()


def test_model(sample, target):
    input_tensor = em(sample)
    input_tensor = torch.reshape(input_tensor, (batch_size, sample.size(0), input_size)).to(device)
    prediction = model(input_tensor)
    #print(f'prediction test: {prediction}')
    #m = nn.Softmax(dim=1)
    #sotfmax_output = m(prediction)
    #print(f'softmax output for test: {sotfmax_output}')
    loss_test = criterion(prediction, target)
    guess = torch.argmax(prediction, dim=1)
    #print(f'guess: {guess}, target: {target}')

    return guess, loss_test.item()



