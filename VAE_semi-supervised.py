import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn import svm
from torchsummary import summary


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_Categorical(x, theta, num_classes):
    x_one_hot = nn.functional.one_hot(
        x.flatten(start_dim=1, end_dim=-1).long(), num_classes=num_classes)
    log_p = torch.sum(
        x_one_hot * torch.log(theta), dim=-1)
    return log_p


def log_standard_Categorical(x):
    x_one_hot = nn.functional.softmax(torch.ones_like(x), dim=1)
    log_p = -torch.sum(x * torch.log(x_one_hot + 1e-8), dim=1)
    return log_p


def log_Normal(x, mu, log_var):
    D = x.shape[1]
    log_p = -.5 * ((x - mu) ** 2. * torch.exp(-log_var) +
                   log_var + D * np.log(2 * np.pi))
    return log_p


def log_standard_Normal(x):
    D = x.shape[1]
    log_p = -.5 * (x ** 2. + D * np.log(2 * np.pi))
    return log_p


class encoder(nn.Module):
    def __init__(self, input_dim, middel_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            32 * self.input_dim * self.input_dim, middel_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, middel_dim, channels, pixel_range):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.pixel_range = pixel_range

        self.input = nn.Linear(
            middel_dim, 32 * self.input_dim * self.input_dim)
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.fully_connected = nn.Linear(
            channels * self.input_dim * self.input_dim, channels * self.input_dim * self.input_dim * self.pixel_range)
        self.softmax = nn.Softmax(dim=2)

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
        x = x.view(-1, self.channels * self.input_dim *
                   self.input_dim, self.pixel_range)
        x = self.softmax(x)
        return x


class classifier(nn.Module):
    def __init__(self, classes, input_dim=68, channels=3):
        super(classifier, self).__init__()
        self.classes = classes
        self.input_dim = input_dim
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, 8, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            16 * self.input_dim * self.input_dim, self.classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 16 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        x = self.softmax(x)
        return x


class Semi_supervised_VAE(nn.Module):
    def __init__(self, classes, pixel_range, latent_dim, input_dim, channels):
        super(Semi_supervised_VAE, self).__init__()

        self.pixel_range = pixel_range
        self.latent_dim = latent_dim
        self.classes = classes
        self.alpha = 0.1
        self.middel_dim = 32 * input_dim * input_dim

        self.eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)

        self.encoder = encoder(input_dim, self.middel_dim, channels)
        self.decoder = decoder(
            input_dim, self.middel_dim, channels, pixel_range)
        self.classifier = classifier(classes, input_dim, channels)

        self.middel_1 = nn.Linear(self.middel_dim + classes, latent_dim)
        self.middel_2 = nn.Linear(self.middel_dim + classes, latent_dim)
        self.middel_3 = nn.Linear(latent_dim + classes, self.middel_dim)

    def encode(self, x, y_hat):
        m = self.encoder.forward(x)
        mu = self.middel_1(torch.cat((m, y_hat), dim=1))
        log_var = self.middel_2(torch.cat((m, y_hat), dim=1))
        return mu, log_var

    def reparameterization(self, mu, log_var):
        return mu + torch.exp(0.5*log_var) * self.eps

    def decode(self, z, y_hat):
        z = self.middel_3(torch.cat((z, y_hat), dim=1))
        return self.decoder.forward(z)

    def classify(self, x):
        return self.classifier.forward(x)

    def ELBO_unlabelled(self, y_prob, log_prior, log_posterior, log_like):
        H = - torch.sum(torch.log(y_prob) * y_prob, dim=-1).mean()
        K_1 = - np.log(1 / self.classes)

        reconstruction_error = - torch.sum(log_like, dim=-1)
        KL = - torch.sum(log_prior - log_posterior, dim=-1)

        Lxy = reconstruction_error + KL + K_1
        q_Lxy = torch.sum(y_prob * Lxy.view(-1, 1), dim=-1).mean()

        ELBO = q_Lxy + H

        reconstruction_error = reconstruction_error.mean()
        KL = KL.mean()

        return ELBO, reconstruction_error, KL

    def ELBO_labelled(self, y, y_prob, log_prior, log_posterior, log_like):
        K_1 = - np.log(1 / self.classes)
        # K_1 = -log_standard_Categorical(y).mean
        log_y = - torch.sum(y * torch.log(y_prob), dim=-1).mean()

        reconstruction_error = - torch.sum(log_like, dim=-1).mean()
        KL = - torch.sum(log_prior - log_posterior, dim=-1).mean()

        Lxy = reconstruction_error + KL + K_1

        ELBO = Lxy + self.alpha * log_y

        return ELBO, reconstruction_error, KL

    def forward(self, x, y):
        y_onehot = nn.functional.one_hot(
            y.long(), num_classes=self.classes).float()

        idx = y != 0  # "DMSO"

        y_labelled = y_onehot[idx]

        y_hat = self.classify(x)
        mu, log_var = self.encode(x, y_hat)

        z = self.reparameterization(mu, log_var)

        theta = self.decode(z, y_hat)

        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)
        log_like = log_Categorical(x, theta, self.pixel_range)

        y_prob_unlabelled = y_hat[~idx]
        y_prob_labelled = y_hat[idx]

        log_posterior_unlabelled = log_posterior[~idx]
        log_posterior_labelled = log_posterior[idx]

        log_prior_unlabelled = log_prior[~idx]
        log_prior_labelled = log_prior[idx]

        log_like_unlabelled = log_like[~idx]
        log_like_labelled = log_like[idx]

        if y_prob_unlabelled.shape[0] == 0:
            ELBO_labelled, reconstruction_error_labelled, KL_labelled = self.ELBO_labelled(
                y_labelled, y_prob_labelled, log_prior_labelled, log_posterior_labelled, log_like_labelled)
            ELBO_unlabelled, reconstruction_error_unlabelled, KL_unlabelled = 0, 0, 0

        elif y_prob_labelled.shape[0] == 0:
            ELBO_unlabelled, reconstruction_error_unlabelled, KL_unlabelled = self.ELBO_unlabelled(
                y_prob_unlabelled, log_prior_unlabelled, log_posterior_unlabelled, log_like_unlabelled)
            ELBO_labelled, reconstruction_error_labelled, KL_labelled = 0, 0, 0

        else:
            ELBO_unlabelled, reconstruction_error_unlabelled, KL_unlabelled = self.ELBO_unlabelled(
                y_prob_unlabelled, log_prior_unlabelled, log_posterior_unlabelled, log_like_unlabelled)
            ELBO_labelled, reconstruction_error_labelled, KL_labelled = self.ELBO_labelled(
                y_labelled, y_prob_labelled, log_prior_labelled, log_posterior_labelled, log_like_labelled)

        reconstruction_error = reconstruction_error_labelled + \
            reconstruction_error_unlabelled
        KL = KL_labelled + KL_unlabelled

        ELBO = ELBO_unlabelled + ELBO_labelled

        tqdm.write(
            f"ELBO: {ELBO.item()}, Reconstruction error: {reconstruction_error.item()}, Regularizer: {KL.item()}")

        return ELBO, reconstruction_error, KL

    def initialise(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.sparse_(m.weight, sparsity=1)
                if m.bias is not None:
                    m.bias.data.fill_(3)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.dirac_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(_init_weights)

    def train_VAE(self, dataloader, epochs, lr=10e-6):
        parameters = [param for param in self.parameters()
                      if param.requires_grad == True]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        self.train()
        self.initialise()

        ELBOs = []
        reconstruction_errors = []
        KLs = []

        self.alpha = 0.1 * dataloader.__len__()
        for epoch in tqdm(range(epochs)):
            for x_batch, y_batch in tqdm(dataloader):
                x = x_batch.to(device)
                y = y_batch.to(device)

                optimizer.zero_grad()

                ELBO, reconstruction_error, KL = self.forward(x, y)

                ELBOs.append(ELBO.item())
                reconstruction_errors.append(reconstruction_error.item())
                KLs.append(KL.item())

                ELBO.backward()
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {ELBO.item()}, Reconstruction Error: {reconstruction_error.item()}, Regularizer: {KL.item()}")

        mu, log_var = self.encode(x)
        latent_space = self.reparameterization(mu, log_var)

        return self.encoder, self.decoder, reconstruction_errors, KL, latent_space


latent_dim = 50
epochs = 5
batch_size = 10

pixel_range = 256
input_dim = 28
channels = 1
classes = 10

train_size = 1000
test_size = 1000

trainset = datasets.MNIST(
    root='./MNIST', train=True, download=True, transform=None)
testset = datasets.MNIST(
    root='./MNIST', train=False, download=True, transform=None)
Xy_train = TensorDataset(trainset.data[:train_size].reshape(
    (train_size, channels, input_dim, input_dim)).float(), trainset.targets[:train_size])
Xy_train = DataLoader(Xy_train, batch_size=batch_size, shuffle=True)
Xy_test = TensorDataset(testset.data[:test_size].reshape(
    (test_size, channels, input_dim, input_dim)).float(), testset.targets[:test_size])
Xy_test = DataLoader(Xy_test, batch_size=batch_size, shuffle=True)

# X = np.load("image_matrix.npz")["images"][:1000]
# X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)

VAE = Semi_supervised_VAE(classes=classes, pixel_range=pixel_range,
                          latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)

print("VAE:")
# The summary function from torchsummary.py has been modified to work with the code by changing the following lines (ln 101):
# total_input_size = abs(np.prod([np.prod(in_size) for in_size in input_size]) * batch_size * 4. / (1024 ** 2.))
# summary(VAE, input_size=[(batch_size, channels, input_dim, input_dim), (batch_size,)])

encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
    dataloader=Xy_train, epochs=epochs)
