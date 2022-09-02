
import VAE
import torchvision
import torch
import matplotlib.pyplot as plt
import pathlib


def gen_new_data(vae, dataset_type, n_samples, vae_loss_type, weights_directory, results_directory):
    if vae_loss_type == "mse":
        BETAS = VAE.MSE_BETAS
    else:
        BETAS = VAE.PERCEPTUAL_BETAS

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle(f'New generated data - {vae_loss_type}', fontsize=40)
    for i in range(len(BETAS)):
        # load checkpoint
        fname = weights_directory / vae_loss_type / ('vae_' + dataset_type + f'_beta_{BETAS[i]}.pth')
        vae.load_state_dict(torch.load(fname))
        # sample from the vae
        vae_samples = vae.sample(num_samples=n_samples).data.cpu().numpy()
        for sample_idx in range(vae_samples.shape[0]):
            ax = fig.add_subplot(len(BETAS), n_samples, sample_idx + 1 + i * n_samples)
            ax.imshow(vae_samples[sample_idx].transpose(1, 2, 0))
            ax.set_axis_off()
            ax.set_title(f'beta_{BETAS[i]}')
    save_path = results_directory / vae_loss_type / (dataset_type + "_new_generated_data.png")
    fig.savefig(save_path)


def main():
    # dataset_type = "cifar10"
    dataset_type = "svhn"
    weights_directory = pathlib.Path("/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints")
    results_directory = pathlib.Path("/home/user_115/Project/Results/Milestone1/VAE_new_generated_data")
    device = "cpu"
    vae = VAE.Vae(x_dim=VAE.X_DIM, in_channels=VAE.INPUT_CHANNELS, z_dim=VAE.Z_DIM).to(device)
    n_samples = 5
    gen_new_data(vae, dataset_type, n_samples, "mse", weights_directory, results_directory)
    gen_new_data(vae, dataset_type, n_samples, "perceptual", weights_directory, results_directory)


if __name__ == '__main__':
    main()
