import VAE
import time
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import PerceptualLossNetwork
import scipy


# define hyper-parameters
BATCH_SIZE = 128  # usually 32/64/128/256
LEARNING_RATE = 2e-4  # for the gradient optimizer
NUM_EPOCHS = 200  # how many epochs to run?


class TrainResults:
    def __init__(self):
        self.recon_errors = []
        self.kl_errors = []
        self.losses = []

    def plot_results(self, results_directory, vae_loss_type, beta, recon_y_label, kl_y_label, loss_y_label):
        # prepare plots params
        recon_save_path, recon_title = get_plot_params(results_directory, vae_loss_type, beta, recon_y_label)
        kl_save_path, kl_title = get_plot_params(results_directory, vae_loss_type, beta, kl_y_label)
        loss_save_path, loss_title = get_plot_params(results_directory, vae_loss_type, beta, loss_y_label)
        # create plots
        create_plot(self.recon_errors, recon_save_path, recon_y_label, recon_title)
        create_plot(self.kl_errors, kl_save_path, kl_y_label, kl_title)
        create_plot(self.losses, loss_save_path, loss_y_label, loss_title)


def train(dataset_name, dataloader, loss_type, beta, train_results, perceptual_loss_network):
    # check if there is gpu available, if there is, use it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running calculations on: ", device)
    # create our model and send it to the device (cpu/gpu)
    vae = VAE.Vae(x_dim=VAE.X_DIM, in_channels=VAE.INPUT_CHANNELS, z_dim=VAE.Z_DIM).to(device)
    # optimizer & scheduler
    vae_optim = torch.optim.Adam(params=vae.parameters(), lr=LEARNING_RATE)
    vae_sched = torch.optim.lr_scheduler.MultiStepLR(vae_optim, milestones=[100, 150], gamma=0.1)
    # save the losses from each epoch, we might want to plot it later
    for epoch in range(NUM_EPOCHS):
        print("epoch #", epoch)
        epoch_start_time = time.time()
        batch_recon_errors = []
        batch_kl_errors = []
        batch_losses = []
        for batch_i, batch in enumerate(dataloader):
            # print("batch #", batch_i)
            # forward pass
            x = batch[0].to(device).view(-1, 3, VAE.X_DIM, VAE.X_DIM)
            x_recon, mu, logvar = vae.forward(x)
            # calculate the loss
            recon_error, kl, loss = vae.loss_function(x_recon, x, mu, logvar, beta, loss_type, perceptual_loss_network)
            # optimization (same 3 steps everytime)
            vae_optim.zero_grad()
            loss.backward()
            vae_optim.step()
            vae_sched.step()
            # save loss
            batch_recon_errors.append(recon_error.data.cpu().item())
            batch_kl_errors.append(kl.data.cpu().item())
            batch_losses.append(loss.data.cpu().item())
        train_results.recon_errors.append(np.mean(batch_recon_errors))
        train_results.kl_errors.append(np.mean(batch_kl_errors))
        train_results.losses.append(np.mean(batch_losses))
        print("epoch: {} training loss: {:.5f} epoch time: {:.3f} sec".format(epoch, train_results.losses[-1],
                                                                              time.time() - epoch_start_time))
        # saving our model (so we don't have to train it again...)

        fname = "./VAE_training_checkpoints/" + loss_type + "/vae_" + dataset_name + "_beta_" + str(beta) + ".pth"
        torch.save(vae.state_dict(), fname)
        print("saved checkpoint @", fname)


def create_plot(y_vals, save_path, ylabel, title):
    x = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(10, 10))
    plt.plot(x, y_vals)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()


def get_plot_params(results_directory, vae_loss_type, beta, ylabel):
    save_path = results_directory / vae_loss_type / (ylabel + "_beta_" + str(beta) + '.png')
    title = save_path.stem
    return save_path, title


def main():
    # in order to create batches of the data, we create a Dataset and a DataLoader, which takes care of:
    # 1. pre-processing the images to tensors with values in [0,1]
    # 2. shuffling the data, so we add randomness as learned in ML
    # 3. if the data size is not divisible by the batch size, we can drop the last batch
    # (so the batches are always of the same size)
    # define pre-procesing transformation to use
    # dataset_name = "cifar10"
    dataset_name = "svhn"
    transform = torchvision.transforms.ToTensor()
    if dataset_name == "cifar10":
        train_data = torchvision.datasets.CIFAR10('./datasets/', train=True, transform=transform,
                                                  target_transform=None, download=True)
    else:
        train_data = torchvision.datasets.SVHN('./datasets/', split="train", transform=transform,
                                               target_transform=None, download=True)

    sample_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    results_directory = pathlib.Path("/home/user_115/Project/Results/Milestone1/VAE_training_plots/")
    recon_y_label = "Reconstruction_error"
    kl_y_label = "KL_error"
    loss_y_label = "Loss"
    # here we go
    loss_types = ["mse", "perceptual"]
    # loss_types = ["perceptual"]
    perceptual_loss_network = None
    for loss_type in loss_types:
        print("loss type: ", loss_type)
        if loss_type == "mse":
            BETAS = VAE.MSE_BETAS
        else:
            perceptual_loss_network = PerceptualLossNetwork.PerceptualLossNetwork()
            perceptual_loss_network.eval()
            BETAS = VAE.PERCEPTUAL_BETAS
        for beta in BETAS:
            train_results = TrainResults()
            train(dataset_name, sample_dataloader, loss_type, beta, train_results,perceptual_loss_network)
            train_results.plot_results(results_directory, loss_type, beta, recon_y_label, kl_y_label, loss_y_label)


if __name__ == '__main__':
    main()
