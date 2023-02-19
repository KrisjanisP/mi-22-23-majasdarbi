import os
import pickle
import time
import matplotlib
import sys
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 7) # size of window
plt.style.use('dark_background')

LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def standartize(data):
    return (data - np.mean(data)) / np.std(data)

class Dataset:
    def __init__(self):
        super().__init__()
        path_dataset = './data/cardekho_india_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('./data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/cardekho_india_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            self.X, self.Y, self.labels = pickle.load(fp)

        self.X = np.array(self.X, dtype=np.float32)
        self.X = standartize(self.X)

        self.Y = np.array(self.Y, dtype=np.float32)
        self.Y = standartize(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.array(self.X[idx]), np.array(self.Y[idx])

class DataLoader:
    def __init__(
            self,
            dataset,
            idx_start, idx_end,
            batch_size
    ):
        super().__init__()
        self.dataset = dataset
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.batch_size = batch_size
        self.idx_batch = 0

    def __len__(self):
        return (self.idx_end - self.idx_start - self.batch_size) // self.batch_size

    def __iter__(self):
        self.idx_batch = 0
        return self

    def __next__(self):
        if self.idx_batch >= len(self):
            raise StopIteration
        
        # define start and end indices
        idx_start = self.idx_start + self.idx_batch * self.batch_size
        idx_end = idx_start + self.batch_size

        # sample x and y
        x, y = self.dataset[idx_start:idx_end]

        # increment batch index
        self.idx_batch += 1
        
        return x, y


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

dataloader_train = DataLoader(
    dataset_full,
    idx_start=0,
    idx_end=train_test_split,
    batch_size=BATCH_SIZE
)
dataloader_test = DataLoader(
    dataset_full,
    idx_start=train_test_split,
    idx_end=len(dataset_full),
    batch_size=BATCH_SIZE
)


class Variable:
    def __init__(self, value, grad=None):
        self.value: np.ndarray = value
        self.grad: np.ndarray = np.zeros_like(value)
        if grad is not None:
            self.grad = grad


class LayerLinear:
    def __init__(self, in_features: int, out_features: int):
        self.W = Variable(
            value=np.random.uniform(low=-1.,size=(in_features, out_features)),
            grad=np.zeros(shape=(BATCH_SIZE, in_features, out_features))
        )
        self.b = Variable(
            value=np.zeros(shape=(out_features,)),
            grad=np.zeros(shape=(BATCH_SIZE, out_features))
        )
        self.x = None
        self.output = None

    def forward(self, x):
        self.x = x
        # W.shape = (in_features, out_features)
        # x.shape = (batch_size, in_features, 1)
        # output.shape = (batch_size, out_features, 1)
        self.output = Variable(
            np.squeeze(self.W.value.T @ np.expand_dims(self.x.value, axis=-1), axis=-1) + self.b.value,
        )
        return self.output

    def backward(self):
        # how does linear change as I change something?

        self.b.grad += 1 * self.output.grad
        self.W.grad += np.expand_dims(self.x.value, axis=-1) @ np.expand_dims(self.output.grad, axis=-2)
        self.x.grad += np.squeeze(self.W.value @ np.expand_dims(self.output.grad, axis=-1), axis=-1)

class LayerSigmoid():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x):
        self.x = x
        self.output = Variable(1 / (1 + np.exp(-self.x.value)))
        return self.output

    def backward(self):
        self.x.grad += self.output.value * (1 - self.output.value) * self.output.grad


class LayerReLU:
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x):
        self.x = x
        self.output = Variable(np.maximum(0, self.x.value))
        return self.output

    def backward(self):
        self.x.grad += -1 * (self.x.value < 0) * self.output.grad


class Swish():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x):
        self.x = x
        self.output = Variable(self.x.value * (1 / (1 + np.exp(-self.x.value))))
        return self.output

    def backward(self):
        self.x.grad += self.output.value * (1 + self.x.value * np.exp(-self.x.value)) * self.output.grad

class LossMSE():
    def __init__(self):
        self.y = None
        self.y_prim  = None

    def forward(self, y, y_prim):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean((self.y.value - self.y_prim.value) ** 2)
        return loss

    def backward(self):
        self.y_prim.grad += -2 * (self.y.value - self.y_prim.value)

class LossMAE():
    def __init__(self):
        self.y = None
        self.y_prim = None

    def forward(self, y, y_prim):
        self.y = y
        self.y_prim = y_prim
        loss = np.sum(np.abs(self.y.value - self.y_prim.value)) / len(self.y.value)
        return loss

    def backward(self):
        self.y_prim.grad += -1 * np.sign(self.y.value - self.y_prim.value)

class Model:
    def __init__(self):
        self.layers = [
            LayerLinear(in_features=6, out_features=4),
            LayerReLU(),
            LayerLinear(in_features=4, out_features=4),
            LayerReLU(),
            LayerLinear(in_features=4, out_features=2),
        ]

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        for layer in reversed(self.layers):
            layer.backward()
        pass

    def parameters(self):
        variables = []
        for layer in self.layers:
            if type(layer) == LayerLinear:
                variables.append(layer.W)
                variables.append(layer.b)
        return variables

class OptimizerSGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.value -= self.learning_rate * np.mean(param.grad, axis=0)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)

# For later
class LayerEmbedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.x_indexes = None
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.emb_m = Variable(np.random.random((num_embeddings, embedding_dim)))
        self.output = None

    def forward(self, x):
        self.x_indexes = x.value.squeeze().astype(np.int)
        self.output = Variable(np.array(self.emb_m.value[self.x_indexes, :])) # same as dot product with one-hot encoded X and Emb_w
        return self.output

    def backward(self):
        self.emb_m.grad[self.x_indexes, :] += self.output.grad


def nrmse_metric(y, y_prim):
    return np.sqrt(np.mean((y - y_prim) ** 2)) / np.std(y)

def r2_score(y, y_prim):
    return 1 - np.sum((y - y_prim) ** 2) / np.sum((y - np.mean(y)) ** 2)

model = Model()
optimizer = OptimizerSGD(
    model.parameters(),
    learning_rate=LEARNING_RATE
)
loss_fn = LossMSE()


loss_plot_train = []
loss_plot_test = []
nrmse_plot_test = []
r2_plot_test = []

for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        nrmse_list = []
        r2_list = []
        for x, y in dataloader:

            y_prim = model.forward(Variable(value=x))
            loss = loss_fn.forward(y=Variable(value=y), y_prim=y_prim)
            nrmse = nrmse_metric(y, y_prim.value)
            r2 = r2_score(y, y_prim.value)

            losses.append(loss)
            nrmse_list.append(nrmse)
            r2_list.append(r2)

            if dataloader == dataloader_train:
                loss_fn.backward()
                model.backward()
                optimizer.step()
                optimizer.zero_grad()

        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
        else:
            loss_plot_test.append(np.mean(losses))
            nrmse_plot_test.append(np.mean(nrmse_list))
            r2_plot_test.append(np.mean(r2_list))

    print(f'epoch: {epoch} loss_train: {loss_plot_train[-1]} loss_test: {loss_plot_test[-1]}')

    if epoch % 10 == 0:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax3 = ax1.twinx()
        ax3.plot(nrmse_plot_test, 'y-', label='nrmse')
        ax4 = ax1.twinx()
        ax4.plot(r2_plot_test, 'g-', label='r2')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper center')
        ax3.legend(loc='upper right')
        ax4.legend(loc='lower right')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        plt.show()