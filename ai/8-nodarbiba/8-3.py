from collections import Counter

import torch
import numpy as np
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from torch.hub import download_url_to_file
import os
import pickle
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm
import sklearn.model_selection
import torch.utils.data

matplotlib.use('TkAgg')
plt.rcParams["figure.figsize"] = (15, 5)
plt.style.use('dark_background')

LEARNING_RATE = 1e-4
BATCH_SIZE = 128
MAX_LEN = 200
TRAIN_TEST_SPLIT = 0.7
DEVICE = 'cpu'

# if torch.cuda.is_available():
#     DEVICE = 'cuda'
#     MAX_LEN = 0


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = './data/Fruits28.pkl'
        if not os.path.exists(path_dataset):
            pass
            os.makedirs('./data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1645110979-deep-learning-intro-2022-q1/Fruits28.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            X, Y, self.labels = pickle.load(fp)
        self.Y_idx = Y

        Y_counter = Counter(Y)
        Y_counts = np.array(list(Y_counter.values()))
        self.Y_weights = (1.0 / Y_counts) * np.sum(Y_counts)

        X = torch.from_numpy(np.array(X).astype(np.float32))
        self.X = X.permute(0, 3, 1, 2)
        self.input_size = self.X.size(-1)
        Y = torch.LongTensor(Y)
        self.Y = F.one_hot(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        return x, y


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

idxes_train, idxes_test = sklearn.model_selection.train_test_split(
    np.arange(len(dataset_full)),
    train_size=train_test_split,
    test_size=len(dataset_full) - train_test_split,
    stratify=dataset_full.Y_idx,
    random_state=0
)

# For debugging
if MAX_LEN:
    idxes_train = idxes_train[:MAX_LEN]
    idxes_test = idxes_test[:MAX_LEN]

dataset_train = Subset(dataset_full, idxes_train)
dataset_test = Subset(dataset_full, idxes_test)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_train) % BATCH_SIZE == 1)
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=(len(dataset_test) % BATCH_SIZE == 1)
)

class DenseSubblock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(num_features=in_channels)
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = torch.relu(x)
        out = self.bn.forward(out)
        out = self.conv.forward(out)
        return out


class DenseBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DenseSubblock(in_channels=32, out_channels=32)
        self.block2 = DenseSubblock(in_channels=64, out_channels=32)
        self.block3 = DenseSubblock(in_channels=96, out_channels=32)
        self.block4 = DenseSubblock(in_channels=128, out_channels=32)

    def forward(self, x):
        out1 = self.block1.forward(x)
        out2 = self.block2.forward(torch.concat([x,out1],dim=1))
        out3 = self.block3.forward(torch.concat([x,out1,out2],dim=1))
        out4 = self.block4.forward(torch.concat([x,out1,out2,out3],dim=1))
        return torch.concat([x,out1,out2,out3,out4],dim=1)


class TransitionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3, stride=1, padding=1)

        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)

        self.pool = torch.nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0
        )

    def forward(self, x):
        out = self.conv.forward(x)
        out = F.relu(out)
        out = self.bn.forward(out)
        out = self.pool.forward(out)
        return out


class DenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3,
                                     out_channels=32,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1,
                                     bias=False)

        self.denseblock_1 = DenseBlock()
        self.denseblock_2 = DenseBlock()
        self.denseblock_3 = DenseBlock()

        self.transition_1 = TransitionLayer(in_channels=160, out_channels=32)
        self.transition_2 = TransitionLayer(in_channels=160, out_channels=32)
        self.transition_3 = TransitionLayer(in_channels=160, out_channels=32)

        self.linear = torch.nn.Linear(in_features=288, out_features=len(dataset_full.labels))

    def forward(self, x):
        out = self.conv1.forward(x)

        out = self.denseblock_1.forward(out)
        print(out.shape)
        out = self.transition_1.forward(out)

        out = self.denseblock_2.forward(out)
        out = self.transition_2.forward(out)

        out = self.denseblock_3.forward(out)
        out = self.transition_3.forward(out)

        out = out.view(-1, 32*3*3)
        out = self.linear.forward(out)
        out = F.softmax(out, dim=1)
        return out


# in deeper models using bias makes no difference and only increases execution time
class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)

        self.is_bottleneck = False
        if stride != 1 or in_channels != out_channels:
            self.is_bottleneck = True
            self.shortcut = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )

    def forward(self, x):
        residual = x

        out = self.conv1.forward(x)
        out = F.relu(out)
        out = self.bn1.forward(out)
        out = self.conv2.forward(out)
        if self.is_bottleneck:
            residual = self.shortcut.forward(residual)
        out += residual
        out = F.relu(out)
        out = self.bn2.forward(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels,
                                     out_channels=8,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=8)
        self.identity_block_1 = ResBlock(in_channels=8, out_channels=8)
        self.identity_block_2 = ResBlock(in_channels=8, out_channels=8)

        self.bottleneck_block_1 = ResBlock(in_channels=8,out_channels=16,stride=2)
        self.identity_block_3 = ResBlock(in_channels=16,out_channels=16)

        self.bottleneck_block_2 = ResBlock(in_channels=16, out_channels=32,stride=2)
        self.identity_block_4 = ResBlock(in_channels=32, out_channels=32)

        self.bottleneck_block_3 = ResBlock(in_channels=32, out_channels=64, stride=2)
        self.identity_block_5 = ResBlock(in_channels=64, out_channels=64)

        self.linear = torch.nn.Linear(
            in_features=64,out_features=len(dataset_full.labels))

    def forward(self, x):
        out = self.bn1.forward(F.relu(self.conv1.forward(x)))
        out = self.identity_block_1.forward(out)
        out = self.identity_block_2.forward(out)
        out = self.bottleneck_block_1.forward(out)
        out = self.identity_block_3.forward(out)
        out = self.bottleneck_block_2.forward(out)
        out = self.identity_block_4.forward(out)
        out = self.bottleneck_block_3.forward(out)
        out = self.identity_block_5.forward(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear.forward(out)
        out = torch.softmax(out, dim=-1)
        return out


def print_model_size(model):
    total_param_size = 0
    for name, param in model.named_parameters():
        each_param_size = np.prod(param.size())
        total_param_size += each_param_size
    print(f'model size is {total_param_size}')


model = DenseNet()
print_model_size(model)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):
    for data_loader in [dataloader_train, dataloader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == dataloader_test:
            stage = 'test'

        for x, y in tqdm(data_loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_prim = model.forward(x)
            loss = torch.mean(-y * torch.log(y_prim + 1e-8))

            if data_loader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()
            x = x.cpu()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plt.subplot(121)  # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 10, 11, 12, 16, 17, 18]):
        plt.subplot(3, 6, j)
        color = 'green' if idx_y[i] == idx_y_prim[i] else 'red'
        plt.title(
            f"pred: {dataset_full.labels[idx_y_prim[i]]}\n real: {dataset_full.labels[idx_y[i]]}", color=color)
        plt.imshow(x[i].permute(1, 2, 0))

    plt.tight_layout(pad=0.5)
    plt.draw()
    plt.pause(0.1)
