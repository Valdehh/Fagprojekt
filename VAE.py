import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_Categorical(x, theta, num_classes):
    x_one_hot = nn.functional.one_hot(
        x.flatten(start_dim=1, end_dim=-1).long(), num_classes=num_classes
    )
    log_p = torch.sum(x_one_hot * torch.log(theta), dim=-1)
    return log_p


def log_Mean_Squared_Error(x, theta):
    x = x.flatten(start_dim=1, end_dim=-1)
    log_p = torch.log(((x - theta) ** 2) / x.shape[1])
    return log_p


def negative_log_Likelihood(x, theta, std):
    x = x.flatten(start_dim=1, end_dim=-1)
    log_p = ((x - theta) ** 2) / (2 * std ** 2)
    return log_p


def log_Normal(x, mu, log_var):
    D = x.shape[1]
    log_p = -0.5 * ((x - mu) ** 2.0 * torch.exp(-log_var) +
                    log_var + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


def log_standard_Normal(x):
    D = x.shape[1]
    log_p = -0.5 * (x**2.0 + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            16 * self.input_dim * self.input_dim, 2 * latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, 16 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        x = nn.LeakyReLU(0.01)(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels

        self.input = nn.Linear(
            latent_dim, 16 * self.input_dim * self.input_dim)
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.output = nn.Linear(channels * self.input_dim * self.input_dim,
                                2 * channels * self.input_dim * self.input_dim)

        # self.softmax = nn.Softmax(dim=1)
        # self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, 16, self.input_dim, self.input_dim)
        x = self.conv2(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, self.channels * self.input_dim * self.input_dim)
        x = self.output(x)
        x = nn.LeakyReLU(0.01)(x)
        # x = self.softmax(x) # i dont think this is needed.. but maybe?
        return x


class VAE(nn.Module):
    def __init__(self, pixel_range, latent_dim, input_dim, channels):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels)
        self.pixel_range = pixel_range
        self.latent_dim = latent_dim
        self.channels = channels
        self.input_dim = input_dim

        self.eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
        # self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim))

    def encode(self, x):
        mu, log_var = torch.split(
            self.encoder.forward(x), self.latent_dim, dim=1)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        return mu + torch.exp(0.5 * log_var) * self.eps

    def decode(self, z):
        mu, log_var = torch.split(
            self.decoder.forward(z), self.channels * self.input_dim * self.input_dim, dim=1)
        std = torch.exp(0.5 * log_var)
        return mu, std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)

        decode_mu, decode_std = self.decode(z)
        # decode_std = 0.1 * torch.ones(decode_mu.shape).to(device)
        theta = torch.normal(mean=decode_mu, std=decode_std)

        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)
        log_like = nn.functional.mse_loss(theta, x.flatten(
            start_dim=1, end_dim=-1), reduction="none").sum(-1)
        # log_like = log_Categorical(x, theta, self.pixel_range)

        reconstruction_error = torch.sum(log_like, dim=-1).mean()
        regularizer = - torch.sum(log_prior - log_posterior, dim=-1).mean()

        elbo = reconstruction_error + regularizer

        tqdm.write(
            f"ELBO: {elbo.item()}, Reconstruction error: {reconstruction_error.item()}, Regularizer: {regularizer.item()}")

        return elbo, reconstruction_error, regularizer

    def initialise(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.sparse_(m.weight, sparsity=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(3)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.dirac_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(_init_weights)

    def train_VAE(self, dataloader, epochs, lr=10e-5):
        parameters = [
            param for param in self.parameters() if param.requires_grad == True
        ]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        reconstruction_errors = []
        regularizers = []

        self.initialise()
        self.train()
        for epoch in tqdm(range(epochs)):
            for batch in tqdm(dataloader):
                x = batch.to(device)
                optimizer.zero_grad()
                elbo, reconstruction_error, regularizer = self.forward(x)
                reconstruction_errors.append(
                    reconstruction_error.item())
                regularizers.append(regularizer.item())
                elbo.backward()
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo.item()}, Reconstruction Error: {reconstruction_error.item()}, Regularizer: {regularizer.item()}"
            )

        mu, log_var = self.encode(x)
        latent_space = self.reparameterization(mu, log_var)

        return (
            self.encoder,
            self.decoder,
            reconstruction_errors,
            regularizers,
            latent_space,
        )


def generate_image(X, encoder, decoder, latent_dim, channels, input_dim, batch_size=1):
    encoder.eval()
    decoder.eval()
    X = X.to(device)
    mu, log_var = torch.split(encoder.forward(X), latent_dim, dim=1)
    eps = torch.normal(mean=0, std=torch.ones(latent_dim))
    z = mu + torch.exp(0.5 * log_var) * eps
    theta = decoder.forward(z)
    image = torch.argmax(theta, dim=-1)
    image = image.reshape((batch_size, channels, input_dim, input_dim))
    image = torch.permute(image, (0, 2, 3, 1))
    image = image.numpy()
    return image


latent_dim = 128
epochs = 5
batch_size = 10

pixel_range = 256
input_dim = 68
channels = 3

train_size = 1000
test_size = 1000

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)
"""
trainset = datasets.MNIST(root="./MNIST", train=True,
                          download=True, transform=None)
testset = datasets.MNIST(root="./MNIST", train=False,
                         download=True, transform=None)
X_train = DataLoader(
    trainset.data[:train_size]
    .reshape((train_size, channels, input_dim, input_dim))
    .float(),
    batch_size=batch_size,
    shuffle=True,
)
X_test = DataLoader(
    testset.data[:test_size]
    .reshape((test_size, channels, input_dim, input_dim))
    .float(),
    batch_size=batch_size,
    shuffle=True,
)
"""
X = np.load("image_matrix.npz")["images"][:1000]
X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
X_train = DataLoader(X, batch_size=batch_size, shuffle=True)
X_test = DataLoader(X, batch_size=batch_size, shuffle=True)

VAE = VAE(
    pixel_range=pixel_range,
    latent_dim=latent_dim,
    input_dim=input_dim,
    channels=channels,
).to(device)

# print("VAE:")
# summary(VAE, input_size=(channels, input_dim, input_dim))

encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
    dataloader=X_train, epochs=epochs)

torch.save(encoder_VAE, "encoder_VAE.pt")
torch.save(decoder_VAE, "decoder_VAE.pt")

np.savez("latent_space_VAE.npz", latent_space=latent_space.detach().numpy())

plt.plot(
    np.arange(0, len(reconstruction_errors), 1),
    reconstruction_errors,
    label="Reconstruction Error",
)
plt.plot(np.arange(0, len(regularizers), 1), regularizers, label="Regularizer")
plt.xlabel("Epochs")
plt.xticks(ticks=np.arange(0, train_size * epochs, 1),
           labels=np.arange(0, train_size * epochs, 1), minor=True)
plt.xticks(
    ticks=np.arange(
        0, (epochs + 1) * train_size / batch_size, train_size / batch_size
    ),
    labels=np.arange(0, epochs + 1, 1),
)
plt.title("ELBO Components")
plt.legend()
plt.show()

X_test = iter(X_test)
generated_images = []
for i in range(9):
    image = generate_image(
        next(X_test),
        encoder_VAE,
        decoder=decoder_VAE,
        latent_dim=latent_dim,
        channels=channels,
        input_dim=input_dim,
    )
    generated_images.append(image)

fig, ax = plt.subplots(3, 3)
ax = ax.flatten()
for i, image in enumerate(generated_images):
    ax[i].imshow(image, cmap="gray")
    ax[i].set_xticks([])
    ax[i].set_yticks([])
fig.suptitle("Generated images")
plt.show()
