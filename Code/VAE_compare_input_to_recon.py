import VAE
import VAE_training
import torchvision
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure
import matplotlib.pyplot as plt
import pathlib


def save_ssim(vae, dataset_type, test_data, vae_loss_type, weights_directory, results_directory, BETAS, device):
    ssims = ""

    for beta in BETAS:
        # load checkpoint
        fname = weights_directory / vae_loss_type / ('vae_' + dataset_type + f'_beta_{beta}.pth')
        # vae.load_state_dict(torch.load(fname, map_location="cpu"))
        vae.load_state_dict(torch.load(fname))
        ssim = calc_ssim(vae, test_data, VAE_training.BATCH_SIZE, device)
        ssims += f"beta:{beta} , ssim:{ssim}\n"
    save_path = results_directory / vae_loss_type / (dataset_type + "SSIM.txt")
    with open(save_path, 'w') as file:
        file.write(ssims)


def calc_ssim(vae, test_data, batch_size, device):
    ssim = 0
    num_of_batches = len(test_data) / batch_size
    dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)
    for samples, _ in dataloader:
        samples = samples.to(device)
        recon_samples, _, _ = vae.forward(samples)
        ssim += structural_similarity_index_measure(recon_samples, samples).item()
    return ssim / num_of_batches


def plot_samples_and_recons(vae, dataset_type, input_channels, samples, n_samples, vae_loss_type, weights_directory, results_directory,
                            BETAS, device):

    fig = plt.figure(figsize=(32, 16))
    fig.suptitle(f'Compare input to reconstruction - {vae_loss_type}', fontsize=40)
    for sample_idx in range(n_samples):
        sample = samples[sample_idx].view(input_channels, VAE.X_DIM, VAE.X_DIM)
        ax = fig.add_subplot(len(BETAS) + 1, n_samples, sample_idx + 1)
        ax.imshow(sample.data.cpu().numpy().transpose(1, 2, 0), cmap='gray')
        # ax.imshow(sample.data.cpu().numpy().transpose(1, 2, 0))
        ax.set_axis_off()
        ax.set_title(f'original')
    for i in range(len(BETAS)):
        # load checkpoint
        fname = weights_directory / vae_loss_type / ('vae_' + dataset_type + f'_beta_{BETAS[i]}.pth')
        # vae.load_state_dict(torch.load(fname, map_location="cpu"))
        vae.load_state_dict(torch.load(fname))
        recon_samples, _, _ = vae.forward(samples.to(device))
        for sample_idx in range(n_samples):
            recon_sample = recon_samples[sample_idx].view(input_channels, VAE.X_DIM, VAE.X_DIM)
            ax = fig.add_subplot(len(BETAS) + 1, n_samples, sample_idx + 1 + (i + 1) * n_samples)
            ax.imshow(recon_sample.data.cpu().numpy().transpose(1, 2, 0), cmap='gray')
            # ax.imshow(recon_sample.data.cpu().numpy().transpose(1, 2, 0))
            ax.set_axis_off()
            ax.set_title(f'beta_{BETAS[i]} recon')
    save_path = results_directory / vae_loss_type / (dataset_type + "_compare_input_to_recon.png")
    fig.savefig(save_path)


def main():

    # dataset_type = "cifar10"
    # dataset_type = "svhn"
    # dataset_type = "svhn_extra"
    # dataset_type = "flowers"
    # dataset_type = "cars"
    # dataset_type = "stl10"
    dataset_type = "fashion_mnist"
    input_channels = 3 if dataset_type != "fashion_mnist" else 1
    # loss_types = ["momentum_perceptual", "vgg_perceptual", "mse"]
    loss_types = ["vgg_perceptual"]
    weights_directory = pathlib.Path("/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints")
    results_directory = pathlib.Path("/home/user_115/Project/Results/Milestone1/VAE_compare_input_to_recon")
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running calculations on: ", device)
    vae = VAE.Vae(x_dim=VAE.X_DIM, in_channels=input_channels, z_dim=VAE.Z_DIM).to(device)
    vae.eval()
    n_samples = 5
    test_dataset = VAE_training.get_dataset(dataset_type, is_train=False, split="test")
    dataloader = DataLoader(test_dataset, batch_size=n_samples, shuffle=False, drop_last=True)
    samples, _ = next(iter(dataloader))

    for loss_type in loss_types:
        BETAS = VAE_training.get_betas_by_loss_type(loss_type)
        plot_samples_and_recons(vae, dataset_type, input_channels, samples, n_samples, loss_type, weights_directory, results_directory, BETAS, device)
        save_ssim(vae, dataset_type, test_dataset, loss_type, weights_directory, results_directory, BETAS, device)


if __name__ == '__main__':
    main()
