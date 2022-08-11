import PerceptualLossNetwork
import math
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# re parametrization trick
def reparameterize(mu, logvar, device=torch.device("cpu")):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variance of x
    :param device: device to perform calculations on
    :return z: the sampled latent variable
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


# encoder - q_{\phi}(z|X)
class VaeEncoder(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, x_dim=28 * 28, hidden_size=256, z_dim=10, device=torch.device("cpu")):
        super(VaeEncoder, self).__init__()
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.device = device

        self.features = nn.Sequential(nn.Linear(x_dim, self.hidden_size),
                                      nn.ReLU())

        self.fc1 = nn.Linear(self.hidden_size, self.z_dim, bias=True)  # fully-connected to output mu
        self.fc2 = nn.Linear(self.hidden_size, self.z_dim, bias=True)  # fully-connected to output logvar

    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """
        mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable
        z = reparameterize(mu, logvar, device=self.device)
        return z, mu, logvar

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        z, mu, logvar = VaeEncoder(X)
        """
        h = self.features(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar


# decoder - p_{\theta}(x|z)
class VaeDecoder(torch.nn.Module):
    """
       This class builds the decoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       """

    def __init__(self, x_dim=28 * 28, hidden_size=256, z_dim=10):
        super(VaeDecoder, self).__init__()
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.z_dim = z_dim

        self.decoder = nn.Sequential(nn.Linear(self.z_dim, self.hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_size, self.x_dim),
                                     nn.Sigmoid())
        # why we use sigmoid? becaue the pixel values of images are in [0,1] and sigmoid(x) does just that!

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        x_reconstruction = VaeDecoder(z)
        """
        x = self.decoder(x)
        return x


class Vae(torch.nn.Module):
    def __init__(self, x_dim=28*28, z_dim=10, hidden_size=256, device=torch.device("cpu")):
        super(Vae, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.encoder = VaeEncoder(x_dim, hidden_size, z_dim=z_dim, device=device)
        self.decoder = VaeDecoder(x_dim, hidden_size, z_dim=z_dim)
        self.perceptual_loss_network = None

    def encode(self, x):
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample(self, num_samples=1):
        """
        This functions generates new data by sampling random variables and decoding them.
        Vae.sample() actually generates new data!
        Sample z ~ N(0,1)
        """
        z = torch.randn(num_samples, self.z_dim).to(self.device)
        return self.decode(z)

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        return x_recon, mu, logvar, z = Vae(X)
        """
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar, beta=1, loss_type='bce'):
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
            if self.perceptual_loss_network is None:
                self.perceptual_loss_network = PerceptualLossNetwork.PerceptualLossNetwork()
                self.perceptual_loss_network.eval()
            x = x.reshape((int((x.size(0)/3)), 3, int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1)))))
            recon_x = recon_x.reshape((int((recon_x.size(0)/3)), 3, int(math.sqrt(recon_x.size(1))),
                                       int(math.sqrt(recon_x.size(1)))))
            recon_error = self.perceptual_loss_network.calc_loss(recon_x, x)
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

