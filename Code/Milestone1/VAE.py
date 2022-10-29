# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import VAE_training

HIDDEN_DIMS = [64, 128, 256, 512]  # size of the hidden layers outputs in the networks
INPUT_CHANNELS = 3
X_DIM = 32  # size of the input dimension
Z_DIM = 128  # size of the latent dimension
MOMENTUM_PERCEPTUAL_BETAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
# MOMENTUM_PERCEPTUAL_BETAS = [10, 100, 1000]
VGG_PERCEPTUAL_BETAS = [1, 30, 40, 50]
# VGG_PERCEPTUAL_BETAS = [50]
MSE_BETAS = [1e-3, 1e-2, 1e-1, 1]
# MSE_BETAS = []
WARMUP_BETA = 1e-1


class VaeEncoder(torch.nn.Module):
    def __init__(self, in_channels, z_dim):
        super(VaeEncoder, self).__init__()
        hidden_dims = [64, 64, 128, 128, 256, 256, 512, 512]
        modules = []
        for idx, h_dim in enumerate(hidden_dims):
            if (idx % 2) == 0:
                stride = 2
            else:
                stride = 1
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.layers = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, z_dim)  # fully-connected to output mu
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, z_dim)  # fully-connected to output logvar

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        features = self.layers(input)
        features = torch.flatten(features,
                                 start_dim=1)  # concatenating (channels, width, height) to vector of size channels*width*height
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class VaeDecoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(VaeDecoder, self).__init__()
        hidden_dims = [512, 256, 128, 64]
        # hidden_dims = [512, 512, 256, 256, 128, 128, 64, 64]
        modules = []
        self.first_layer = nn.Linear(z_dim, hidden_dims[0] * 4)
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
                nn.Sigmoid()))
        self.layers = nn.Sequential(*modules)

    def forward(self, z):
        result = self.first_layer(z)
        result = result.view(-1, 512, 2, 2)
        result = self.layers(result)
        return result


class Vae(torch.nn.Module):
    def __init__(self, x_dim=32, in_channels=3, z_dim=128):
        super(Vae, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.encoder = VaeEncoder(in_channels, z_dim)
        self.decoder = VaeDecoder(z_dim)

    def encode(self, input):
        return self.encoder.forward(input)

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, input):
        z, mu, log_var = self.encode(input)
        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mu, logvar, beta, loss_type, perceptual_loss_network):
        """
        This function calculates the loss of the VAE.
        loss = reconstruction_loss - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param recon_x: the reconstruction from the decoder
        :param x: the original input
        :param mu: the mean given X, from the encoder
        :param logvar: the log-variance given X, from the encoder
        :param loss_type: type of loss function - 'mse', 'l1', 'bce'
        :return: VAE loss
        """
        if loss_type == 'mse':
            recon_error = F.mse_loss(recon_x, x, reduction='sum')
        elif loss_type == 'l1':
            recon_error = F.l1_loss(recon_x, x, reduction='sum')
        elif loss_type == 'bce':
            recon_error = F.binary_cross_entropy(recon_x, x, reduction='sum')
        elif loss_type == 'vgg_perceptual':
            recon_error = perceptual_loss_network.calc_loss(recon_x, x)
            # recon_error = perceptual_loss_network(recon_x,x, reduction='sum')
        elif loss_type == 'momentum_perceptual':
            recon_error = perceptual_loss_network.calc_loss(recon_x, x)
        else:
            raise NotImplementedError

        recon_error = recon_error / x.size(0)  # normalize by batch_size
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # normalize by batch_size
        total_loss = recon_error + beta * kl
        return recon_error, kl, total_loss

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.z_dim)
        return self.decode(z)


def get_betas_by_loss_type(vae_loss_type):
    if vae_loss_type == "mse":
        return MSE_BETAS
    elif vae_loss_type == "vgg_perceptual":
        return VGG_PERCEPTUAL_BETAS
    else:
        return MOMENTUM_PERCEPTUAL_BETAS
