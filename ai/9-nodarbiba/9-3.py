import os
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import torchvision
from torch.hub import download_url_to_file
from tqdm import tqdm # pip install tqdm
import random

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15, 10)
plt.style.use('dark_background')

import torch.utils.data
import scipy.misc
import scipy.ndimage
import sklearn.decomposition

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
TRAIN_TEST_SPLIT = 0.8
DEVICE = 'cuda'
MAX_LEN = 0

if True or not torch.cuda.is_available():
    MAX_LEN = 26 # limit max number of samples otherwise too slow training (on GPU use all samples / for final training)
    DEVICE = 'cpu'
    BATCH_SIZE = 6

class DatasetApples(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = './data/apples_dataset.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('./data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1630528570-intro-course-2021-q4/apples_dataset.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            X, Y, self.labels = pickle.load(fp)

        X = torch.from_numpy(np.array(X)).float()
        self.X = X.permute(0, 3, 1, 2)
        self.input_size = self.X.size(-1)
        Y = torch.LongTensor(Y)
        self.Y = Y.unsqueeze(dim=-1)
        self.Training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] / 255.0
        y_label = self.Y[idx]
        y_target = x.clone()
        noise = torch.randn(x.size()[1:])
        noise = torch.concat([noise, noise, noise])
        noise = noise.reshape(x.size())
        x[noise<0] = 1
        return x, y_target, y_label

dataset_full = DatasetApples()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(1)
)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_train) % BATCH_SIZE == 1)
)

data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_test) % BATCH_SIZE == 1)
)


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=7, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=7, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),

            torch.nn.AdaptiveAvgPool2d(output_size=(1,1))
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(size=(9,9)),

            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=1),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=7, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=4),
            torch.nn.LeakyReLU(),

            torch.nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=8, stride=1, padding=0),
            torch.nn.BatchNorm2d(num_features=3),
            torch.nn.Sigmoid(),
        )


    def forward(self, x):
        z = self.encoder.forward(x)
        z = z.view(-1, 32)
        y_prim = self.decoder.forward(z.view(-1, 32, 1, 1))
        return y_prim, z


model = AutoEncoder()

dummy = torch.randn((BATCH_SIZE, 3, 100, 100))
y_target = model.forward(dummy)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(DEVICE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'z',
        'labels'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):

    metrics_epoch = {key: [] for key in metrics.keys()}

    for data_loader in [data_loader_train, data_loader_test]:
        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'
            model = model.eval()
            dataset_full.eval()
        else:
            model = model.train()
            dataset_full.train()

        for x, y_target, y_label in tqdm(data_loader, desc=stage):
            x = x.to(DEVICE)
            y_target = y_target.to(DEVICE)
            y_label = y_label.squeeze().to(DEVICE)

            y_prim, z = model.forward(x)
            loss = torch.mean(torch.abs(y_prim-y_target))
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_z = z.cpu().data.numpy()
            np_x = x.cpu().data.numpy()
            np_y_target = y_target.cpu().data.numpy()
            np_y_label = y_label.cpu().data.numpy()

            metrics_epoch[f'{stage}_z'] += np_z.tolist()
            metrics_epoch[f'{stage}_labels'] += np_y_label.tolist()

    metrics_strs = []
    for key in metrics_epoch.keys():
        if '_z' not in key and '_labels' not in key:
            value = np.mean(metrics_epoch[key])
            metrics[key].append(value)
            metrics_strs.append(f'{key}: {round(value, 2)}')
    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.subplot(221) # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        value = scipy.ndimage.gaussian_filter1d(value, sigma=2)

        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 16, 17, 18]):
        plt.subplot(8, 6, j) # row col idx
        plt.title(f"class: {dataset_full.labels[np_y_label[i]]}")
        plt.imshow(np.transpose(np_x[i], (1, 2, 0)))

        plt.subplot(8, 6, j+6) # row col idx
        plt.imshow(np.transpose(np_y_prim[i], (1, 2, 0)))

    plt.subplot(223) # row col idx

    pca = sklearn.decomposition.KernelPCA(n_components=2, gamma=0.1)

    plt.title('train_z')
    np_z = np.array(metrics_epoch[f'train_z'])
    np_z = pca.fit_transform(np_z)
    np_z_label = np.array(metrics_epoch[f'train_labels'])
    scatter = plt.scatter(np_z[:, -1], np_z[:, -2], c=np_z_label)
    plt.legend(handles=scatter.legend_elements()[0], labels=dataset_full.labels)

    plt.subplot(224) # row col idx

    plt.title('test_z')
    np_z = np.array(metrics_epoch[f'test_z'])
    np_z = pca.fit_transform(np_z)
    np_z_label = np.array(metrics_epoch[f'test_labels'])
    scatter = plt.scatter(np_z[:, -1], np_z[:, -2], c=np_z_label)
    plt.legend(handles=scatter.legend_elements()[0], labels=dataset_full.labels)

    plt.tight_layout(pad=0.5)
    plt.savefig('results.png')
    plt.clf()
    plt.cla()
