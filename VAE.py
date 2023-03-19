import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class encoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super(self, nn.Module).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(32 * 68 * 68, 2 * latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * 68 * 68)
        x = self.fully_connected(x)
        return x
    

class decoder(nn.Module):
    def __init__(self, latent_dim):
        super(self, nn.Module).__init__()
        self.input = nn.Linear(latent_dim, 32 * 68 * 68) 
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=2),
        self.conv2 = nn.ConvTranspose2d(
            16, 3, kernel_size=5, stride=1, padding=2)
        self.fully_connected = nn.Linear(3 * 68 * 68, 3 * 68 * 68 * 256)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32, 68, 68)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 3 * 32 * 32)
        x = self.fully_connected(x)
        x = x.view(-1, 3 * 68 * 68, 256)
        x = self.softmax(x)
        return x


class VAE():
    def __init__(self, X, latent_dim):
        self.encoder = encoder(latent_dim)
        self.decoder = decoder(latent_dim)

        self.data_length = len(X)
        self.prior = torch.normal(mean = 0, std = torch.ones(self.data_length))
        
    def encode(self, x):
        mu, std = torch.split(self.encoder.forward(x), 2, dim=1)
        return mu, std

    def reparameterization(self, mu, log_var, eps):
        return mu + torch.exp(0.5*log_var) * eps 

    def decode(self, z):
        return self.decoder.forward(z)

    def ELBO(self, x):
        eps = torch.normal(mean = 0, std = torch.ones(len(x)))
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var, eps)
        theta = self.decode(z)
        raise NotImplementedError
        posterior = torch.normal(mean = mu, std = torch.exp(0.5 * log_var))
        con_like = 1
        elbo = -(torch.log(con_like) + torch.log(posterior) - torch.log(self.prior))
        