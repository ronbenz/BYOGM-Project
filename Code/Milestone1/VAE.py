import math
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIMS = [64, 128, 256, 512]  # size of the hidden layers outputs in the networks
INPUT_CHANNELS = 3
X_DIM = 32  # size of the input dimension
Z_DIM = 128  # size of the latent dimension
MSE_BETAS = [1e-3, 1e-2, 1e-1, 1]
# MSE_BETAS = []
PERCEPTUAL_BETAS = [1, 30, 40, 50]
# PERCEPTUAL_BETAS = [40, 50]


class Vae(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       """

    def __init__(self, x_dim=32, in_channels=3, z_dim=128):
        super(Vae, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        hidden_dims = [64, 128, 256, 512]
        modules = []
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, z_dim)  # fully-connected to output mu
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, z_dim) # fully-connected to output logvar
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(z_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid()) # why we use sigmoid? becaue the pixel values of images are in [0,1] and sigmoid(x) does just that!

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

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
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
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
        elif loss_type == 'perceptual':
            recon_error = perceptual_loss_network.calc_loss(recon_x, x)
        else:
            raise NotImplementedError

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        recon_error = recon_error / x.size(0)  # normalize by batch_size
        kl = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # normalize by batch_size
        total_loss = recon_error + beta*kl
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


