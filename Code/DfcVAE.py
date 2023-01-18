import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import torchvision.models as models
import torchvision
import VAE_training
import pathlib
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image, ImageFile

BS = 32
lr = 0.0001
epochs = 100
sz = 64
alpha = 1
beta = 0.5
Z_DIM = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

content_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                   'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                   'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                   'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                   'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']


class _VGG(nn.Module):
    '''
    Classic pre-trained VGG19 model.
    Its forward call returns a list of the activations from
    the predefined content layers.
    '''

    def __init__(self):
        super(_VGG, self).__init__()
        features = models.vgg19(pretrained=True).features

        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            self.features.add_module(name, module)

    def forward(self, input):
        batch_size = input.size(0)
        all_outputs = []
        output = input
        for name, module in self.features.named_children():
            output = module(output)
            if name in content_layers:
                all_outputs.append(output.view(BS, -1))
        return all_outputs


class DfcVae(nn.Module):
    def __init__(self, latent_dim=100, leaky_relu=0.2):
        super().__init__()
        self.leaky_relu = leaky_relu
        self.encoder = self.create_encoder(latent_dim)
        # not sure about this FC part tbh
        self.fc = nn.Sequential(nn.Linear(latent_dim, 4096), nn.ReLU(True))
        self.decoder = self.create_decoder()

    def encode(self, x):
        x = self.encoder(x)
        return x[:, :100], x[:, 100:]

    def decode(self, z):
        z = self.decoder(z)
        return z.view(-1, 3 * 64 * 64)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # todo: delete ".cpu()"
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        z = self.fc(z)
        z = z.view(-1, 256, 4, 4)
        x_bar = self.decoder(z)
        return x_bar, mu, logvar

    def create_conv_block(self, in_c, out_c, stride=2, kernel=(4, 4), padding=1):
        return [
            nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(self.leaky_relu, True)
        ]

    def create_encoder(self, latent_dim):
        encoder_as_list = self.create_conv_block(3, 32)
        encoder_as_list += self.create_conv_block(32, 64)
        encoder_as_list += self.create_conv_block(64, 128)
        encoder_as_list += self.create_conv_block(128, 256)
        encoder_as_list += [nn.Flatten(), nn.Linear(256 * 4 * 4, latent_dim * 2)]
        return nn.Sequential(*encoder_as_list)

    def create_decode_block(self, in_c, out_c, stride=1, kernel=(3, 3), scale=2, final=False):
        block = [
            nn.UpsamplingNearest2d(scale_factor=scale),
            nn.Conv2d(in_c, out_c, kernel, stride=stride, padding_mode='replicate', padding=1)
        ]

        if not final:
            block += [
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU()
            ]
        return block

    def create_decoder(self):
        decoder_as_list = self.create_decode_block(256, 128)
        decoder_as_list += self.create_decode_block(128, 64)
        decoder_as_list += self.create_decode_block(64, 32)
        decoder_as_list += self.create_decode_block(32, 3, final=True)
        return nn.Sequential(*decoder_as_list)

    def sample(self, num_samples):
        #dist = torch.distributions.normal.Normal(mu, std)
        z = torch.randn(num_samples, Z_DIM).to(torch.device(device))
        z = self.fc(z)
        z = z.view(-1, 256, 4, 4)
        z = self.decoder(z)
        return z


class KLDLoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(KLDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, mean, logvar):
        # KLD loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), 1)
        # Size average
        if self.reduction == 'mean':
            kld_loss = torch.mean(kld_loss)
        elif self.reduction == 'sum':
            kld_loss = torch.sum(kld_loss)
        return kld_loss


class FLPLoss(nn.Module):
    def __init__(self, device, reduction='sum'):
        super(FLPLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.pretrained = _VGG().to(device)

    def forward(self, x, recon_x):
        x_f = self.pretrained(x)
        recon_f = self.pretrained(recon_x)
        return self._fpl(recon_f, x_f)

    def _fpl(self, recon_f, x_f):
        fpl = 0
        for _r, _x in zip(recon_f, x_f):
            fpl += self.criterion(_r, _x)
        return fpl


def load_model(model_construct, path=None):
    model = model_construct()
    if path is not None:
        model.load_state_dict(torch.load(path))
    return model


def train(model, name, dataloader, device, train_results):
    print("running calculations on: ", device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    calc_kld = KLDLoss()
    calc_flp = FLPLoss(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        data_num = 0
        batch_recon_errors = []
        batch_kl_errors = []
        batch_losses = []
        for batch_idx, data in enumerate(dataloader):
            data = data[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)

            # Loss
            kld_loss = calc_kld(mu, logvar)
            flp_loss = calc_flp(data, recon_x)
            total_loss = alpha * flp_loss + beta * kld_loss

            # Train
            total_loss.backward()
            optimizer.step()

            # save loss
            batch_recon_errors.append(flp_loss.data.cpu().item())
            batch_kl_errors.append(kld_loss.data.cpu().item())
            batch_losses.append(total_loss.data.cpu().item())

            # Metrics
            running_loss += total_loss
            data_num += data.size(0)
            # if batch_idx % 25 == 0:
            # loss_data = total_loss.data
            # print(
            #       'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tVGG: {:.6f}\tKLD: {:.6f}'.
            #       format(epoch, batch_idx * len(data), len(dataloader.dataset),
            #         100. * batch_idx / len(dataloader), loss_data, flp_loss, kld_loss))

        epoch_loss = running_loss / data_num
        train_results.recon_errors.append(np.mean(batch_recon_errors))
        train_results.kl_errors.append(np.mean(batch_kl_errors))
        train_results.losses.append(np.mean(batch_losses))
        print("Epoch: {} has Nanway loss {:.6f} our loss {:.6f}".format(epoch, epoch_loss, train_results.losses[-1]))

        # for validation in [to_validate, to_validate_two, to_validate_three, to_validate_four]:
        #     test_on_images(model, validation)


def generate_images(mu, std, model, inv_normalize, n_imgs=1):
    dist = torch.distributions.normal.Normal(mu, std)

    fig = plt.figure(figsize=(25, 16))
    for i in range(n_imgs):
        random = dist.sample()
        random = random.unsqueeze(0)
        recon = model.sample(random)

        ax = fig.add_subplot(1, n_imgs, i + 1, xticks=[], yticks=[])
        plt.imshow((inv_normalize(recon[0]).detach().cpu().numpy().transpose(1, 2, 0)))

    plt.show()


def main():
    # train description
    loss_type = "Dfc"
    dataset_name = "svhn"
    recon_y_label = "Reconstruction_error"
    kl_y_label = "KL_error"
    loss_y_label = "Loss"

    results_directory = pathlib.Path("/home/user_115/Project/Results/Milestone1/VAE_training_plots/")
    weights_directory = pathlib.Path("/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints")
    root = '/home/user_115/Project/Code/Milestone1/datasets/'
    transform = transforms.Compose([transforms.Resize(sz), transforms.CenterCrop(sz), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255], std=[1 / 0.229, 1 / 0.224, 1 / 0.255])

    dataset = torchvision.datasets.SVHN(root=root, split="train", transform=transform, target_transform=None, download=True)
    dataloader = DataLoader(dataset, batch_size=BS, shuffle=True)

    model = load_model(DfcVae)
    model.to(device)

    train_metrics = VAE_training.TrainMetrics()
    train(model, "vgg", dataloader, device, train_metrics)
    # saving our model
    model_fname = "/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints/" + loss_type + "/vae_" + dataset_name + "_beta_" + str(
        beta) + ".pth"

    torch.save(model.state_dict(), model_fname)
    print("saved checkpoint @", model_fname)


    # comapre_input_to_recon and new_generated_data
    model.load_state_dict(torch.load(model_fname))
    print("calculating metrics")

    model.eval()
    test_dataset = torchvision.datasets.SVHN(root=root, split="test", transform=transform, target_transform=None, download=True)
    VAE_training.report_train_metrics(model, test_dataset, dataloader, BS, train_metrics, device)
    train_metrics.plot_results(results_directory, loss_type, beta, recon_y_label, kl_y_label, loss_y_label, dataset_name, epochs)
    metrics_str = f"beta:{beta}, fid:{train_metrics.fid}, ssim: {train_metrics.ssim}\n"
    VAE_training.save_train_metrics(metrics_str, results_directory, loss_type, dataset_name, epochs)


if __name__ == '__main__':
    main()
