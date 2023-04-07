import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_Categorical(x, theta, num_classes):
    x_one_hot = nn.functional.one_hot(
        x.flatten().long(), num_classes=num_classes)
    log_p = torch.sum(
        x_one_hot * torch.log(torch.clamp(theta, 10e-8, 1.-10e-8)), dim=-1)
    return log_p


def log_Normal(x, mu, log_var):
    D = x.shape[0]
    log_p = -.5 * ((x - mu) ** 2. * torch.exp(-log_var) +
                   log_var + D * np.log(2 * np.pi))
    return log_p


def log_standard_Normal(x):
    D = x.shape[0]
    log_p = -.5 * (x ** 2. + D * np.log(2 * np.pi))
    return log_p


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            32 * self.input_dim * self.input_dim, 2 * latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.input = nn.Linear(
            latent_dim, 32 * self.input_dim * self.input_dim)
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.fully_connected = nn.Linear(
            channels * self.input_dim * self.input_dim, channels * self.input_dim * self.input_dim * 256)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32, self.input_dim, self.input_dim)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, self.channels * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        x = x.view(-1, self.channels * self.input_dim * self.input_dim, 256)
        x = self.softmax(x)
        return x


class VAE(nn.Module):
    def __init__(self, X, pixel_range, latent_dim, input_dim, channels):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels)
        self.pixel_range = pixel_range
        self.latent_dim = latent_dim

        self.data_length = len(X)
        self.eps = torch.normal(mean=0, std=torch.ones(latent_dim))
        # self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim))

    def encode(self, x):
        mu, std = torch.split(self.encoder.forward(x), self.latent_dim, dim=1)
        return mu, std

    def reparameterization(self, mu, log_var):
        return mu + torch.exp(0.5*log_var) * self.eps

    def decode(self, z):
        return self.decoder.forward(z)

    def ELBO(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)
        theta = self.decode(z)
        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)
        log_con_like = log_Categorical(x, theta, self.pixel_range)
        reconstruction_error = torch.sum(log_con_like, dim=-1)
        regularizer = - torch.sum(log_posterior - log_prior, dim=-1)
        elbo = - torch.mean(reconstruction_error + regularizer)
        print(elbo, reconstruction_error, regularizer)
        return elbo, reconstruction_error, regularizer

    def train_VAE(self, X, epochs, batch_size, lr=10e-10):
        parameters = [param for param in self.parameters()
                      if param.requires_grad == True]
        optimizer = torch.optim.SGD(parameters, lr=lr)
        reconstruction_errors = []
        regularizers = []
        for epoch in tqdm(range(epochs)):
            for i in tqdm(range(0, self.data_length, batch_size)):
                x = X[i:i+batch_size].to(device)
                optimizer.zero_grad()
                elbo, reconstruction_error, regularizer = self.ELBO(x)
                reconstruction_errors.append(reconstruction_error)
                regularizers.append(regularizer)
                elbo.backward(retain_graph=True)
                optimizer.step()
            if epochs == epoch + 1:
                latent_space = self.reparameterization(self.encode(x))
            print(
                f"Epoch: {epoch+1}, ELBO: {elbo}, Reconstruction Error: {reconstruction_error}, Regularizer: {regularizer}")
        return self.encoder, self.decoder, reconstruction_errors, regularizers, latent_space


def generate_image(decoder, latent_dim, output_dim, channels):
    z = torch.normal(mean=0, std=torch.ones(1, latent_dim))
    theta = decoder.forward(z)
    theta = theta[0].detach().numpy()
    theta = theta.argmax(axis=1)
    theta = theta.reshape((channels, output_dim, output_dim))
    return theta


latent_dim = 5
epochs = 5
batch_size = 1

trainset = torchvision.datasets.MNIST(
    root='./MNIST', train=True, download=True, transform=None)
X = trainset.data.numpy()
X = torch.tensor(X[:1000], dtype=torch.float32)

# X = np.load("image_matrix.npz")["images"][:1000]

pixel_range = 256
input_dim = 28
channels = 1

VAE = VAE(X, pixel_range=pixel_range,
          latent_dim=latent_dim, input_dim=input_dim, channels=channels)
encoder, decoder, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
    X=X, epochs=epochs, batch_size=batch_size)
torch.save(encoder, "encoder.pt")
torch.save(decoder, "decoder.pt")

np.savez("latent_space.npz", latent_space=latent_space)

plt.plot(reconstruction_errors, label="Reconstruction Error")
plt.plot(regularizers, label="Regularizer")
plt.xlabel("Epochs")
plt.xticks(ticks=np.arange(0, epochs*len(X), len(X)),
           labels=np.arange(0, epochs, 1))
plt.title("ELBO Components")
plt.legend()
plt.show()

output_dim = 28

generated_images = []
for _ in range(9):
    image = generate_image(
        decoder=decoder, latent_dim=latent_dim, output_dim=output_dim, chanels=channels)
    generated_images.append(image)

fig, ax = plt.subplots(3, 3)
ax = ax.flatten()
for i, image in enumerate(generated_images):
    ax[i].imshow(image)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.title("Generated images")
plt.show()
