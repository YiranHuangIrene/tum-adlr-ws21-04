import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_log_var = nn.Linear(hidden_size, latent_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class TrajectoryVAE(nn.Module):
    def __init__(self, input_dim, hidden_size,latent_size):
        super(TrajectoryVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_size,hidden_size)
        self.decoder = Decoder(latent_size, hidden_size,input_dim)

    def reparametrize(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.rand_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparametrize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


class Trajectory(Dataset,):
    def __init__(self,length = None):
        self.data = torch.tensor(np.load('trajectory.npy'))
        self.length = length
    def __len__(self):
        return len(self.data) if self.length == None else self.length

    def __getitem__(self, item):
        return self.data[item, :]
