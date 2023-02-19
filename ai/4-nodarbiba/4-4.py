import matplotlib.pyplot as plt
import numpy as np

X = np.array([[5,3.2],[8,3.8],[6,3.0],[9,1.45]]) # + 2000, * 100000
Y = np.array([3.2,6,3.7,5.6]) # * 1000
# https://www.ss.lv/msg/lv/transport/cars/audi/a6/fgfcf.html
# https://www.ss.lv/msg/lv/transport/cars/audi/a6/hhlkj.html
# https://www.ss.lv/msg/lv/transport/cars/audi/a6/cnohi.html
# https://www.ss.lv/msg/lv/transport/cars/audi/a6/fhfem.html

W_1 = 0
b_1 = 0
W_2 = 0
b_2 = 0
W_3 = 0
b_3 = 0

def linear(W, b, x):
    return W*x+b

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def model(x, W_1, b_1, W_2, b_2, W_3, b_3):
    layer_1 = linear(W_1, b_1, x)
    layer_2 = tanh(layer_1)
    layer_3 = linear(W_2, b_2, layer_2)
    layer_4 = tanh(layer_3)
    layer_5 = linear(W_3, b_3, layer_4)
    return layer_5

def loss_mae(y_prim, y):
    return np.sum(np.abs(y_prim-y))

def loss_mse(y_prim, y):
    return np.sum((y_prim-y)**2)

def dW_linear(W, b, x):
    return x

def db_linear(W, b, x):
    return 1

def dx_linear(W, b, x):
    return W

def dy_prim_loss_mae(y_prim, y):
    return (y_prim-y)/(np.abs(y_prim-y)+1e-8)

def dW_1_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_1 = dW_linear(W_1, b_1, x)
    d_layer_2 = dx_sigmoid(linear(W_1, b_1, x))
    d_layer_3 = dx_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))
    d_loss = dy_prim_loss_mae(y_prim, y)
    return d_loss*d_layer_3*d_layer_2*d_layer_1

def db_1_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_1 = db_linear(W_1, b_1, x)
    d_layer_2 = dx_sigmoid(linear(W_1, b_1, x))
    d_layer_3 = dx_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))
    d_loss = dy_prim_loss_mae(y_prim, y)
    return d_loss*d_layer_3*d_layer_2*d_layer_1

def dW_2_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_3 = dW_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))
    d_loss = dy_prim_loss_mae(y_prim, y)
    return d_loss*d_layer_3

def db_2_loss(x, W_1, b_1, W_2, b_2, y_prim, y):
    d_layer_3 = db_linear(W_2, b_2, sigmoid(linear(W_1, b_1, x)))
    d_loss = dy_prim_loss_mae(y_prim, y)
    return d_loss*d_layer_3


learning_rate = 1e-4
losses = []
for epoch in range(1000000):

    Y_prim = model(X, W_1, b_1, W_2, b_2)
    loss = loss_mae(Y_prim, Y)

    W_1 = W_1 - learning_rate * np.sum(dW_1_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))
    b_1 = b_1 - learning_rate * np.sum(db_1_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))
    W_2 = W_2 - learning_rate * np.sum(dW_2_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))
    b_2 = b_2 - learning_rate * np.sum(db_2_loss(X, W_1, b_1, W_2, b_2, Y_prim, Y))

    losses.append(loss)

print(f'Y_prim: {Y_prim}')
print(f'Y: {Y}')
print(f'loss: {loss}')

plt.plot(losses)
plt.show()