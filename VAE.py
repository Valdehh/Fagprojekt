import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_Categorical(x, theta, num_classes):
    x_one_hot = nn.functional.one_hot(x.long(), num_classes=num_classes)
    log_p = torch.log(theta)*x_one_hot
    return torch.sum(log_p, dim=1)

class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(encoder, self).__init__()
        self.input_dim=input_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(32 * self.input_dim * self.input_dim, 2 * latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        return x
    

class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(decoder, self).__init__()
        self.input_dim=input_dim
        self.input = nn.Linear(latent_dim, 32 * self.input_dim * self.input_dim) 
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.conv2 = nn.ConvTranspose2d(
            16, 3, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.fully_connected = nn.Linear(3 * self.input_dim * self.input_dim, 3 * self.input_dim * self.input_dim * 256)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32, self.input_dim, self.input_dim)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 3 * 32 * 32)
        x = self.fully_connected(x)
        x = x.view(-1, 3 * self.input_dim * self.input_dim, 256)
        x = self.softmax(x)
        return x


class VAE():
    def __init__(self, X, pixel_range, latent_dim):
        self.encoder = encoder(latent_dim)
        self.decoder = decoder(latent_dim)
        self.pixel_range = pixel_range

        self.data_length = len(X)
        self.prior = torch.distributions.MultivariateNormal(loc = torch.zeros(latent_dim), covariance_matrix = torch.eye(latent_dim))
        
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
        log_posterior = torch.sum(torch.log(torch.distributions.MultivariateNormal(loc = mu, covariance_matrix = torch.diag(torch.exp(log_var)))), dim=1)
        log_prior = torch.sum(torch.log(self.prior), dim=1)
        log_con_like = log_Categorical(x, theta, self.pixel_range)
        reconstruction_error = torch.mean(log_con_like)
        regularizer = torch.mean(log_posterior - log_prior)
        elbo = -(reconstruction_error + regularizer)
        return elbo, reconstruction_error, regularizer


    def train(self, X, epochs, batch_size, lr=0.001):
        optimizer_encoder = torch.optim.SGD(self.encoder.parameters(), lr=lr)
        optimizer_decoder = torch.optim.SGD(self.decoder.parameters(), lr=lr)
        reconstruction_errors = []
        regularizers = []
        for epoch in range(epochs):
            for i in range(0, self.data_length, batch_size):
                x = X[i:i+batch_size].to(device)
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                elbo, reconstruction_error, regularizer = self.ELBO(x)
                reconstruction_errors.append(reconstruction_error)
                regularizers.append(regularizer)
                elbo.backward()
                optimizer_encoder.step()
                optimizer_decoder.step()
            print(f"Epoch: {epoch}, ELBO: {elbo}, Reconstruction Error: {reconstruction_error}, Regularizer: {regularizer}")
        return self.encoder, self.decoder, reconstruction_errors, regularizers



def generate_image(decoder, latent_dim, output_dim):
    z = torch.normal(mean = 0, std = torch.ones(1, latent_dim))
    theta = decoder.forward(z)
    theta = theta.argmax(dim=1)
    theta = theta.view(3, output_dim, output_dim)
    theta = theta.detach().numpy()
    plt.imshow(theta)
    plt.show()