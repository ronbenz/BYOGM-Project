
import VAE
import time
from torch.utils.data import DataLoader, Dataset
from fid_score import calc_fid_from_dataset_generate
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import VggPerceptualLossNetwork
import MomentumEncoder
import pathlib


# define hyper-parameters
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
NUM_EPOCHS = 400


class TrainResults:
    def __init__(self):
        self.recon_errors = []
        self.kl_errors = []
        self.losses = []
        self.fid = None

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
            loss_trace_str += f"#epoch:{epoch+1} ,loss:{loss:.2f}\n"
        with open(file_path, 'w') as file:
            file.write(loss_trace_str)

    def plot_results(self, results_directory, vae_loss_type, beta, recon_y_label, kl_y_label, loss_y_label, dataset_name):
        # prepare plots params
        plot_save_path, plot_title = self.get_file_params(results_directory, vae_loss_type, beta, dataset_name, '.png')
        loss_trace_save_path, _ = self.get_file_params(results_directory, vae_loss_type, beta, dataset_name, '.txt')
        self.create_plot(plot_save_path, recon_y_label, kl_y_label, loss_y_label, plot_title)
        self.create_loss_trace(loss_trace_save_path)


def train(dataset_name, dataloader, vae, loss_type, num_of_epochs, learning_rate, beta, device, train_results, perceptual_loss_network, use_pretraining_weights=False):

    if use_pretraining_weights:
        initialize_weights(vae, perceptual_loss_network, loss_type, dataset_name)

    # optimizer & scheduler
    vae_optim = torch.optim.Adam(params=vae.parameters(), lr=learning_rate)
    #vae_sched = torch.optim.lr_scheduler.MultiStepLR(vae_optim, milestones=[int(num_of_epochs*0.5), int(num_of_epochs*0.75)], gamma=0.1)
    vae_sched = torch.optim.lr_scheduler.MultiStepLR(vae_optim, milestones=[num_of_epochs-50], gamma=0.1)
    #vae_sched = torch.optim.lr_scheduler.MultiStepLR(vae_optim, milestones=[150], gamma=0.1)
    #vae_sched = torch.optim.lr_scheduler.MultiStepLR(vae_optim, milestones=[200, 250], gamma=0.1)
    num_images_for_fid_calc = len(dataloader.dataset) - (len(dataloader.dataset)%BATCH_SIZE)
    # save the losses from each epoch, we might want to plot it later
    for epoch in range(num_of_epochs):
        print(f"epoch #{epoch+1}")
        epoch_start_time = time.time()
        batch_recon_errors = []
        batch_kl_errors = []
        batch_losses = []

        for batch_i, batch in enumerate(dataloader):
            # print("batch #", batch_i)
            # forward pass
            x = batch[0].to(device).view(-1, VAE.INPUT_CHANNELS, VAE.X_DIM, VAE.X_DIM)
            x_recon, mu, logvar = vae.forward(x)
            # calculate the loss
            recon_error, kl, loss = vae.loss_function(x_recon, x, mu, logvar, beta, loss_type, perceptual_loss_network)
            # optimization (same 3 steps everytime)
            vae_optim.zero_grad()
            loss.backward()
            vae_optim.step()
            # save loss
            batch_recon_errors.append(recon_error.data.cpu().item())
            batch_kl_errors.append(kl.data.cpu().item())
            batch_losses.append(loss.data.cpu().item())
            if loss_type == "momentum_perceptual" and ((batch_i+1) % 10 == 0):
                perceptual_loss_network.update_target_weights(vae.encoder)

        train_results.recon_errors.append(np.mean(batch_recon_errors))
        train_results.kl_errors.append(np.mean(batch_kl_errors))
        train_results.losses.append(np.mean(batch_losses))
        # MultiStepLR
        vae_sched.step()

        print("epoch:{} training loss: {:.5f} epoch time: {:.3f} sec".format(epoch+1, train_results.losses[-1], time.time() - epoch_start_time))

    # report fid
    fid = calc_fid_from_dataset_generate(dataloader, vae, BATCH_SIZE, cuda=True, dims=2048, device=device,
                                         num_images=min(num_images_for_fid_calc, 50000))
    train_results.fid = fid
    #saving our model (so we don't have to train it again...)
    vae_fname = "/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints/" + loss_type + "/vae_" + dataset_name + "_beta_" + str(beta) + ".pth"
    torch.save(vae.state_dict(), vae_fname)
    print("saved checkpoint @", vae_fname)
    #momentum_encoder_fname = "./VAE_training_checkpoints/" + loss_type + "/momentum_encoder_" + dataset_name + "_beta_" + str(beta) + ".pth"
    #torch.save(perceptual_loss_network.encoder.state_dict(), momentum_encoder_fname)
    #print("saved checkpoint @", momentum_encoder_fname)


def save_fids(fids_str,results_directory,loss_type,dataset_name):
    fid_file_path = results_directory / f"{loss_type}" / f"{dataset_name}_fids_after_{NUM_EPOCHS}_epochs.txt"
    with open(fid_file_path, 'w') as ffp:
        ffp.write(fids_str)


def initialize_weights(vae, perceptual_loss_network, loss_type,dataset_name):
    vae_f_name = f"/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints/starting_weights/{loss_type}/vae_mse_beta_0.1_" + dataset_name + ".pth"
    vae.load_state_dict(torch.load(vae_f_name))
    if loss_type == 'momentum_perceptual':
        momentum_encoder_f_name = f"/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints/starting_weights/{loss_type}/momentum_encoder_mse_beta_0.1_" + dataset_name + ".pth"
        perceptual_loss_network.encoder.load_state_dict(torch.load(momentum_encoder_f_name))


def main():
    # in order to create batches of the data, we create a Dataset and a DataLoader, which takes care of:
    # 1. pre-processing the images to tensors with values in [0,1]
    # 2. shuffling the data, so we add randomness as learned in ML
    # 3. if the data size is not divisible by the batch size, we can drop the last batch
    # (so the batches are always of the same size)
    # define pre-procesing transformation to use
    # dataset_name = "cifar10"
    dataset_name = "svhn"
    # dataset_name = "flowers"
    # dataset_name = "stl10"
    # dataset_name = "fashion_mnist"
    transform = torchvision.transforms.ToTensor()
    resize_transform = torchvision.transforms.Compose([transform, torchvision.transforms.Resize((VAE.X_DIM, VAE.X_DIM), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)])
    root = '/home/user_115/Project/Code/Milestone1/datasets/'
    # resize_transform = torchvision.transforms.Compose([transform, torchvision.transforms.CenterCrop(VAE.X_DIM)])
    if dataset_name == "cifar10":
        train_data = torchvision.datasets.CIFAR10(root, train=True, transform=transform,
                                                  target_transform=None, download=True)
    elif dataset_name == "svhn":
        train_data = torchvision.datasets.SVHN(root, split="train", transform=transform,
                                               target_transform=None, download=True)
    elif dataset_name == "flowers":
        train_data = torchvision.datasets.Flowers102(root, split="train", transform=resize_transform,
                                                     target_transform=None, download=True)
    elif dataset_name == "cars":
        train_data = torchvision.datasets.StanfordCars(root, split="train", transform=resize_transform,
                                                       target_transform=None, download=True)
    elif dataset_name == "stl10":
        # test set has more images, so we use it as train
        train_data = torchvision.datasets.STL10(root, split="test", transform=resize_transform,
                                                       target_transform=None, download=True)
    else:
        train_data = torchvision.datasets.FashionMNIST(root, train=True, transform=resize_transform,
                                                       target_transform=None, download=True)

    sample_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    results_directory = pathlib.Path("/home/user_115/Project/Results/Milestone1/VAE_training_plots/")
    recon_y_label = "Reconstruction_error"
    kl_y_label = "KL_error"
    loss_y_label = "Loss"
    # here we go
    loss_types = ["momentum_perceptual", "mse", "vgg_perceptual"]
    # loss_types = ["vgg_perceptual"]
    perceptual_loss_network = None
    for loss_type in loss_types:
        print("loss type: ", loss_type)
        if loss_type == "mse":
            BETAS = VAE.MSE_BETAS
        elif loss_type == "vgg_perceptual":
            perceptual_loss_network = VggPerceptualLossNetwork.VggPerceptualLossNetwork()
            perceptual_loss_network.eval()
            perceptual_loss_network.requires_grad_(False)
            # perceptual_loss_network = VggPerceptualLossNetworkTalVersion.VGGDistance(device=device)
            BETAS = VAE.VGG_PERCEPTUAL_BETAS
        else:
            BETAS = VAE.MOMENTUM_PERCEPTUAL_BETAS

        fids_str = ""
        for beta in BETAS:
            # check if there is gpu available, if there is, use it
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #device = torch.device("cpu")
            print("running calculations on: ", device)
            # create our model and send it to the device (cpu/gpu)
            vae = VAE.Vae(x_dim=VAE.X_DIM, in_channels=VAE.INPUT_CHANNELS, z_dim=VAE.Z_DIM, device=device).to(device)

            if loss_type == "momentum_perceptual":
                # sample layers for 32x32 input images
                perceptual_loss_network = MomentumEncoder.MomentumEncoder(vae.encoder, 0.99, [0, 1, 2, 3])
                # sample layers for 64x64 input images
                # perceptual_loss_network = MomentumEncoder.MomentumEncoder(vae.encoder, 0.99, [0, 1, 2, 3, 4])
                #perceptual_loss_network = MomentumEncoder.MomentumEncoder(vae.encoder, 0.99, [1, 3, 5, 7])
                perceptual_loss_network.encoder.requires_grad_(False)
            train_results = TrainResults()
            train(dataset_name, sample_dataloader, vae, loss_type, NUM_EPOCHS, LEARNING_RATE, beta, device, train_results, perceptual_loss_network,  use_pretraining_weights=False)
            train_results.plot_results(results_directory, loss_type, beta, recon_y_label, kl_y_label, loss_y_label, dataset_name)
            fids_str += f"beta:{beta}, fid:{train_results.fid}\n"
        save_fids(fids_str, results_directory,loss_type, dataset_name)




if __name__ == '__main__':
    main()
