import VAE
import time
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib
# define hyper-parameters
BATCH_SIZE = 128  # usually 32/64/128/256
LEARNING_RATE = 1e-3  # for the gradient optimizer
NUM_EPOCHS = 150  # how many epochs to run?
HIDDEN_SIZE = 256  # size of the hidden layers in the networks
X_DIM = 32 * 32  # size of the input dimension
Z_DIM = 10  # size of the latent dimension
BETA = [0.25, 0.5, 0.75, 1, 2, 4, 6, 10]


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


def train(dataloader, loss_type, beta, train_results):
    # check if there is gpu available, if there is, use it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running calculations on: ", device)
    # create our model and send it to the device (cpu/gpu)
    vae = VAE.Vae(x_dim=X_DIM, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, device=device).to(device)
    # optimizer
    vae_optim = torch.optim.Adam(params=vae.parameters(), lr=LEARNING_RATE)
    # save the losses from each epoch, we might want to plot it later
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        batch_recon_errors = []
        batch_kl_errors = []
        batch_losses = []
        for batch_i, batch in enumerate(dataloader):
            # forward pass
            x_test = batch[0].to(device)
            x = batch[0].to(device).view(-1, X_DIM)  # create a matrix with the images as its rows ( 3 rows = 1 RGB image )
            x_recon, mu, logvar, z = vae(x)
            # calculate the loss
            recon_error, kl, loss = vae.loss_function(x_recon, x, mu, logvar, beta, loss_type)
            # optimization (same 3 steps everytime)
            vae_optim.zero_grad()
            loss.backward()
            vae_optim.step()
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
        fname = "./VAE_training_checkpoints/" + loss_type + "/vae_cifar10_" + "beta_" + str(beta) + ".pth"
        torch.save(vae.state_dict(), fname)
        print("saved checkpoint @", fname)


def create_plot(y_vals, save_path, ylabel, title):
    x = range(1, NUM_EPOCHS+1)
    plt.figure(figsize=(10,10)) 
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
    transform = torchvision.transforms.ToTensor()
    train_data = torchvision.datasets.CIFAR10('./datasets/', train=True, transform=transform,
                                            target_transform=None, download=True)
    test_data = torchvision.datasets.CIFAR10('./datasets/', train=False, transform=transform,
                                           target_transform=None, download=True)
    sample_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    results_directory = pathlib.Path("/home/user_115/Project/Results/Milestone1/VAE_training_plots/")
    recon_y_label = "Reconstruction_error"
    kl_y_label = "KL_error"
    loss_y_label = "Loss"
    # here we go
    for beta in BETA:
        train_results_mse = TrainResults()
        train_results_perceptual = TrainResults()
        # train using mse loss
        train(sample_dataloader, "mse", beta, train_results_mse)
        train_results_mse.plot_results(results_directory, "mse", beta, recon_y_label, kl_y_label, loss_y_label)
        # train using perceptual loss
        train(sample_dataloader, "perceptual", beta, train_results_perceptual)
        train_results_perceptual.plot_results(results_directory, "perceptual", beta, recon_y_label, kl_y_label, loss_y_label)


if __name__ == '__main__':
    main()
