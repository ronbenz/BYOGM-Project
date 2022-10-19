import VAE
import time
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import VggPerceptualLossNetwork
import VggPerceptualLossNetworkTalVersion
import MomentumEncoder
import scipy


# define hyper-parameters
BATCH_SIZE = 128  # usually 32/64/128/256
LEARNING_RATE = 2e-4  # for the gradient optimizer
NUM_EPOCHS = 200  # how many epochs to run?
NUM_WARMUP_EPOCHS = 100
# NUM_EPOCHS = 1


class TrainResults:
    def __init__(self):
        self.recon_errors = []
        self.kl_errors = []
        self.losses = []

    def get_file_params(self, results_directory, vae_loss_type, beta, dataset_name, suffix):
        save_path = results_directory / vae_loss_type / (dataset_name + "_beta_" + str(beta) + suffix)
        title = save_path.stem
        return save_path, title

    def create_plot(self, save_path, recon_y_label, kl_y_label, loss_y_label, title):
        fig = plt.figure(figsize=(32, 16))
        fig.suptitle(title, fontsize=40)
        x = range(1, NUM_EPOCHS + 1)

        ax = fig.add_subplot(1, 3, 1)
        ax.plot(x, self.losses)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(loss_y_label)
        ax.set_title(loss_y_label)

        ax = fig.add_subplot(1, 3, 2)
        ax.plot(x, self.recon_errors)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(recon_y_label)
        ax.set_title(recon_y_label)

        ax = fig.add_subplot(1, 3, 3)
        ax.plot(x, self.kl_errors)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(kl_y_label)
        ax.set_title(kl_y_label)

        fig.savefig(save_path)

    def create_loss_trace(self, file_path):
        loss_trace_str = ""
        for epoch, loss in enumerate(self.losses):
            loss_trace_str += f"#epoch:{epoch} ,loss:{loss:.2f} \n"
        with open(file_path, 'w') as file:
            file.write(loss_trace_str)

    def plot_results(self, results_directory, vae_loss_type, beta, recon_y_label, kl_y_label, loss_y_label, dataset_name):
        # prepare plots params
        plot_save_path, plot_title = self.get_file_params(results_directory, vae_loss_type, beta, dataset_name, '.png')
        loss_trace_save_path, _ = self.get_file_params(results_directory, vae_loss_type, beta, dataset_name, '.txt')
        self.create_plot(plot_save_path, recon_y_label, kl_y_label, loss_y_label, plot_title)
        self.create_loss_trace(loss_trace_save_path)


def train(dataset_name, dataloader, vae, loss_type, beta, device, train_results, perceptual_loss_network):

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
            recon_error, kl, loss = vae.loss_function(x_recon, x, mu, logvar, beta, loss_type, perceptual_loss_network, epoch)
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
        # MultiStepLR
        vae_sched.step()
        if loss_type == "momentum_perceptual":
            perceptual_loss_network.update_target_weights(vae.encoder)

        print("epoch: {} training loss: {:.5f} epoch time: {:.3f} sec".format(epoch, train_results.losses[-1],
                                                                              time.time() - epoch_start_time))
        # saving our model (so we don't have to train it again...)

        fname = "./VAE_training_checkpoints/" + loss_type + "/vae_" + dataset_name + "_beta_" + str(beta) + ".pth"
        torch.save(vae.state_dict(), fname)
        print("saved checkpoint @", fname)


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
    #loss_types = ["momentum_perceptual", "vgg_perceptual", "mse"]
    loss_types = ["momentum_perceptual"]
    perceptual_loss_network = None
    for loss_type in loss_types:
        print("loss type: ", loss_type)
        if loss_type == "mse":
            BETAS = VAE.MSE_BETAS
        elif loss_type == "vgg_perceptual":
            perceptual_loss_network = VggPerceptualLossNetwork.VggPerceptualLossNetwork()
            perceptual_loss_network.eval()
            # perceptual_loss_network = VggPerceptualLossNetworkTalVersion.VGGDistance(device=device)
            BETAS = VAE.VGG_PERCEPTUAL_BETAS
        else:
            BETAS = VAE.MOMENTUM_PERCEPTUAL_BETAS

        for beta in BETAS:
            # check if there is gpu available, if there is, use it
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #device = torch.device("cpu")
            print("running calculations on: ", device)
            # create our model and send it to the device (cpu/gpu)
            vae = VAE.Vae(x_dim=VAE.X_DIM, in_channels=VAE.INPUT_CHANNELS, z_dim=VAE.Z_DIM).to(device)

            if loss_type == "momentum_perceptual":
                perceptual_loss_network = MomentumEncoder.MomentumEncoder(vae.encoder, 0.99, [1, 3, 5, 7])
                perceptual_loss_network.encoder.eval()
            train_results = TrainResults()
            train(dataset_name, sample_dataloader, vae, loss_type, beta, device, train_results, perceptual_loss_network)
            train_results.plot_results(results_directory, loss_type, beta, recon_y_label, kl_y_label, loss_y_label, dataset_name)


if __name__ == '__main__':
    main()
