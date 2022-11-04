import VAE
import VAE_training
import torchvision
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure
import matplotlib.pyplot as plt
import pathlib


def calc_ssim(vae, dataset_type, test_data, vae_loss_type, weights_directory, results_directory, BETAS, device):
    ssims = ""
    num_of_batches = len(test_data) / VAE_training.BATCH_SIZE
    for beta in BETAS:
        # load checkpoint
        ssim = 0
        fname = weights_directory / vae_loss_type / ('vae_' + dataset_type + f'_beta_{beta}.pth')
        #vae.load_state_dict(torch.load(fname, map_location="cpu"))
        vae.load_state_dict(torch.load(fname))
        dataloader = DataLoader(test_data, batch_size=VAE_training.BATCH_SIZE, drop_last=True)
        for samples, _ in dataloader:
            samples = samples.to(device)
            recon_samples, _, _ = vae.forward(samples)
            ssim += structural_similarity_index_measure(recon_samples, samples).item()
        ssims += f"beta:{beta} , ssim:{ssim/num_of_batches}\n"
    save_path = results_directory / vae_loss_type / (dataset_type + "SSIM.txt")
    with open(save_path, 'w') as file:
        file.write(ssims)


def plot_samples_and_recons(vae, dataset_type, n_samples, samples, vae_loss_type, weights_directory, results_directory, BETAS, device):
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Compare input to reconstruction - {vae_loss_type}', fontsize=40)
    for sample_idx in range(n_samples):
        sample = samples[sample_idx].view(3, VAE.X_DIM, VAE.X_DIM)
        ax = fig.add_subplot(len(BETAS) + 1, n_samples, sample_idx + 1)
        ax.imshow(sample.cpu().numpy().transpose(1, 2, 0))
        ax.set_axis_off()
        ax.set_title(f'original')
    for i in range(len(BETAS)):
        # load checkpoint
        fname = weights_directory / vae_loss_type / ('vae_' + dataset_type + f'_beta_{BETAS[i]}.pth')
        #vae.load_state_dict(torch.load(fname, map_location="cpu"))
        vae.load_state_dict(torch.load(fname))
        recon_samples, _, _ = vae.forward(samples.to(device))
        for sample_idx in range(n_samples):
            recon_sample = recon_samples[sample_idx].view(3, VAE.X_DIM, VAE.X_DIM)
            ax = fig.add_subplot(len(BETAS) + 1, n_samples, sample_idx + 1 + (i + 1) * n_samples)
            ax.imshow(recon_sample.data.cpu().numpy().transpose(1, 2, 0))
            ax.set_axis_off()
            ax.set_title(f'beta_{BETAS[i]} recon')
    save_path = results_directory / vae_loss_type / (dataset_type + "_compare_input_to_recon.png")
    fig.savefig(save_path)


def main():
    # dataset_type = "cifar10"
    #dataset_type = "svhn"
    dataset_type = "flowers"
    loss_types = ["momentum_perceptual","mse", "vgg_perceptual"]
    #loss_types = ["momentum_perceptual"]
    weights_directory = pathlib.Path("/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints")
    results_directory = pathlib.Path("/home/user_115/Project/Results/Milestone1/VAE_compare_input_to_recon")
    #device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("running calculations on: ", device)
    vae = VAE.Vae(x_dim=VAE.X_DIM, in_channels=VAE.INPUT_CHANNELS, z_dim=VAE.Z_DIM).to(device)
    vae.eval()
    n_samples = 5
    transform = torchvision.transforms.ToTensor()
    if dataset_type == "cifar10":
        test_data = torchvision.datasets.CIFAR10('./datasets/', train=False, transform=transform,
                                                 target_transform=None, download=True)
    elif dataset_type == "svhn":
        test_data = torchvision.datasets.SVHN('./datasets/', split="test", transform=transform,
                                              target_transform=None, download=True)
    else:
        resize_transform = torchvision.transforms.Compose([transform, torchvision.transforms.Resize((VAE.X_DIM, VAE.X_DIM), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)])
        # resize_transform = torchvision.transforms.Compose([transform, torchvision.transforms.CenterCrop(VAE.X_DIM)])
        test_data = torchvision.datasets.Flowers102('./datasets/', split="test", transform=resize_transform, target_transform=None, download=True)
    dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=True, drop_last=True)
    samples, _ = next(iter(dataloader))
    for loss_type in loss_types:
        BETAS = VAE.get_betas_by_loss_type(loss_type)
        plot_samples_and_recons(vae, dataset_type, n_samples, samples, loss_type, weights_directory, results_directory, BETAS, device)
        calc_ssim(vae, dataset_type, test_data, loss_type, weights_directory, results_directory, BETAS, device)


if __name__ == '__main__':
    main()
