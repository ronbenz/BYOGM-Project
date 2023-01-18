import numpy as np

import VAE_training
import VAE
import torchvision
import torch
import matplotlib.pyplot as plt
import pathlib
from torch.utils.data import DataLoader
from fid_score import calc_fid_from_dataset_generate


def calc_fid(vae, dataset_type, vae_loss_type, weights_directory, results_directory, BETAS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running calculations on: ", device)
    train_dataset = VAE_training.get_dataset(dataset_type, is_train=True, split="train")

    fid_results = ""
    for beta in BETAS:
        fname = weights_directory / vae_loss_type / ('vae_' + dataset_type + f'_beta_{beta}.pth')
        vae.load_state_dict(torch.load(fname))
        dataloader = DataLoader(train_dataset, batch_size=VAE_training.BATCH_SIZE,shuffle=True)
        fid = calc_fid_from_dataset_generate(dataloader, vae, VAE_training.BATCH_SIZE, cuda=True, dims=2048, device=device, num_images=min(len(train_dataset), 50000))
        fid_results += f"beta:{beta} , fid:{fid}\n"
    save_path = results_directory / vae_loss_type / (dataset_type + "FID.txt")
    with open(save_path, 'w') as file:
        file.write(fid_results)


def plot_new_generated_data(vae, dataset_type, n_samples, vae_loss_type, weights_directory, results_directory, BETAS):
    for beta in BETAS:
        fig = plt.figure(figsize=(32, 16))
        # fig.suptitle(f'New generated data - {vae_loss_type}', fontsize=40)
        # for i in range(len(BETAS)):
        #     # load checkpoint
        #     fname = weights_directory / vae_loss_type / ('vae_' + dataset_type + f'_beta_{BETAS[i]}.pth')
        #     vae.load_state_dict(torch.load(fname, map_location="cpu"))
        #     # sample from the vae
        #     vae_samples = vae.sample(num_samples=n_samples).data.cpu().numpy()
        #     for sample_idx in range(vae_samples.shape[0]):
        #         ax = fig.add_subplot(len(BETAS), n_samples, sample_idx + 1 + i * n_samples)
        #         ax.imshow(vae_samples[sample_idx].transpose(1, 2, 0))
        #         ax.set_axis_off()
        #         ax.set_title(f'beta_{BETAS[i]}')

        # code for generating images for only one beta
        ##################################################################################################
        # load checkpoint
        fig.suptitle(f'New generated data - {vae_loss_type}, Beta:{beta}', fontsize=40)
        fname = weights_directory / vae_loss_type / ('vae_' + dataset_type + f'_beta_{beta}.pth')
        vae.load_state_dict(torch.load(fname))
        # sample from the vae
        vae_samples = vae.sample(num_samples=n_samples).data.cpu().numpy()
        for sample_idx in range(vae_samples.shape[0]):
            ax = fig.add_subplot(int(np.sqrt(n_samples)), int(np.sqrt(n_samples)), sample_idx + 1)
            ax.imshow(vae_samples[sample_idx].transpose(1, 2, 0), cmap='gray')
            # ax.imshow(vae_samples[sample_idx].transpose(1, 2, 0))
            ax.set_axis_off()
        ##################################################################################################
        save_path = results_directory / vae_loss_type / (dataset_type + f"_new_generated_data_beta_{beta}.png")
        fig.savefig(save_path)


def main():
    # loss_types = ["momentum_perceptual", "mse"]
    loss_types = ["vgg_perceptual"]
    # dataset_type = "cifar10"
    # dataset_type = "svhn"
    # dataset_type = "svhn_extra"
    # dataset_type = "flowers"
    # dataset_type = "cars"
    # dataset_type = "stl10"
    dataset_type = "fashion_mnist"
    input_channels = 3 if dataset_type != "fashion_mnist" else 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_directory = pathlib.Path("/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints")
    results_directory = pathlib.Path("/home/user_115/Project/Results/Milestone1/VAE_new_generated_data")
    vae = VAE.Vae(x_dim=VAE.X_DIM, in_channels=input_channels, z_dim=VAE.Z_DIM, device=device).to(device)
    vae.eval()
    for loss_type in loss_types:
        BETAS = VAE_training.get_betas_by_loss_type(loss_type)
        n_samples = 25
        plot_new_generated_data(vae, dataset_type, n_samples, loss_type, weights_directory, results_directory, BETAS)
        # calc_fid(vae, dataset_type, loss_type, weights_directory, results_directory, BETAS)


if __name__ == '__main__':
    main()
