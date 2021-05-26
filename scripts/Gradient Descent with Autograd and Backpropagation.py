#Manually : NUMPY
#import numpy as np
import torch

# f = w * x
# f = 2 * x

#X = np.array([1, 2, 3, 4], dtype=np.float32)
#Y = np.array([1, 4, 9, 16], dtype=np.float32)

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([1, 4, 9, 16], dtype=torch.float32)

#w = 0.0
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w * x

# loss
def loss(y, y_predicted):
    return((y-y_predicted)**2).mean()

# gradient
# MSE = 1/N * (w*x-y)**2
# dj/dw = 1/N 2x (wx-y)
#def gradient(x,y,y_predicted):
    #return np.dot(2*x,y_predicted-y).mean()

print('prediction before training {}'.format(forward(5)))


# training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    #gradients = backward pass
    #dw = gradient(X,Y,y_pred)
    l.backward() #dl/dw

    # update weights
    with torch.no_grad(): #just update w and not cal agin
        w -= learning_rate * w.grad

    # zeros gradient before start a new iteration
    w.grad.zero_()


    if epoch % 20 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


print('prediction after training {}'.format(forward(5)))

