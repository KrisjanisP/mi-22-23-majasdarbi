import matplotlib.pyplot as plt
import numpy as np

X = np.array([4,1,0,0,7]) # + 2000
Y = np.array([2.5,1.5,2.2,2.15,5]) # * 1000

W_1 = 0
b_1 = 0
W_2 = 0
b_2 = 0

def linear(W, b, x):
    return W*x+b

def sigmoid(x):
    return 1/(1+np.exp(-x))

def model(x, W_1, b_1, W_2, b_2):
    layer_1 = linear(W_1, b_1, x)
    layer_2 = sigmoid(layer_1)
    layer_3 = linear(W_2, b_2, layer_2)
    return layer_3

def loss_mae(y_prim, y):
    return np.sum(np.abs(y_prim-y))

def dW_linear(W, b, x):
    return 0 #TODO

def db_linear(W, b, x):
    return 0 #TODO

def dx_linear(W, b, x):
    return 0 #TODO

def dx_sigmoid(x):
    return 0 #TODO

def dy_prim_loss_mae(y_prim, y):
    return 0 #TODO

def dW_1_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    return 0 #TODO

def db_1_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    return 0 #TODO

def dW_2_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    return 0 #TODO

def db_2_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    return 0 #TODO


learning_rate = 1e-1
losses = []
for epoch in range(1000):

    Y_prim = model(X, W_1, b_1, W_2, b_2)
    loss = loss_mae(Y_prim, Y)

    #TODO SGD
    dW_1 = np.sum(dW_1_loss(X, W_1, b_1, W_2, b_2, Y_prim, y))

    print(f'Y_prim: {Y_prim}')
    print(f'Y: {Y}')
    print(f'loss: {loss}')
    losses.append(loss)

plt.plot(losses)
plt.show()