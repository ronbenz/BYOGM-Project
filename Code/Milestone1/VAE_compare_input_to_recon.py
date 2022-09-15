import VAE
import torchvision
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pathlib


def plot_samples_and_recons(vae, dataset_type, n_samples, samples, vae_loss_type, weights_directory, results_directory):
    if vae_loss_type == "mse":
        BETAS = VAE.MSE_BETAS
    elif vae_loss_type == "vgg_perceptual":
        BETAS = VAE.VGG_PERCEPTUAL_BETAS
    else:
        BETAS = VAE.MOMENTUM_PERCEPTUAL_BETAS

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Compare input to reconstruction - {vae_loss_type}', fontsize=40)
    for sample_idx in range(n_samples):
        sample = samples[sample_idx].view(3, VAE.X_DIM, VAE.X_DIM)
        ax = fig.add_subplot(len(BETAS) + 1, n_samples, sample_idx + 1)
        ax.imshow(sample.numpy().transpose(1, 2, 0))
        ax.set_axis_off()
        ax.set_title(f'original')
    for i in range(len(BETAS)):
        # load checkpoint
        fname = weights_directory / vae_loss_type / ('vae_' + dataset_type + f'_beta_{BETAS[i]}.pth')
        vae.load_state_dict(torch.load(fname))
        recon_samples, _, _ = vae.forward(samples)
        for sample_idx in range(n_samples):
            recon_sample = recon_samples[sample_idx].view(3, VAE.X_DIM, VAE.X_DIM)
            ax = fig.add_subplot(len(BETAS) + 1, n_samples, sample_idx + 1 + (i + 1) * n_samples)
            ax.imshow(recon_sample.data.numpy().transpose(1, 2, 0))
            ax.set_axis_off()
            ax.set_title(f'beta_{BETAS[i]} recon')
    save_path = results_directory / vae_loss_type / (dataset_type + "_compare_input_to_recon.png")
    fig.savefig(save_path)


def main():
    # dataset_type = "cifar10"
    dataset_type = "svhn"
    weights_directory = pathlib.Path("/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints")
    results_directory = pathlib.Path("/home/user_115/Project/Results/Milestone1/VAE_compare_input_to_recon")
    vae = VAE.Vae(x_dim=VAE.X_DIM, in_channels=VAE.INPUT_CHANNELS, z_dim=VAE.Z_DIM)
    vae.eval()
    n_samples = 5
    transform = torchvision.transforms.ToTensor()
    if dataset_type == "cifar10":
        test_data = torchvision.datasets.CIFAR10('./datasets/', train=False, transform=transform,
                                                 target_transform=None, download=True)
    else:
        test_data = torchvision.datasets.SVHN('./datasets/', split="test", transform=transform,
                                              target_transform=None, download=True)

    sample_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=True, drop_last=True)
    samples, labels = next(iter(sample_dataloader))
    #plot_samples_and_recons(vae, dataset_type, n_samples, samples, "mse", weights_directory, results_directory)
    plot_samples_and_recons(vae, dataset_type, n_samples, samples, "vgg_perceptual", weights_directory,
                            results_directory)
    #plot_samples_and_recons(vae, dataset_type, n_samples, samples, "momentum", weights_directory, results_directory)


if __name__ == '__main__':
    main()
