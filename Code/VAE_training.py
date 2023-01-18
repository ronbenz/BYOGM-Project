import VAE
import VAE_compare_input_to_recon
import VAE_gen_new_data
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

# MOMENTUM_PERCEPTUAL_BETAS = [0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20]
# MOMENTUM_PERCEPTUAL_BETAS = [2]
MOMENTUM_PERCEPTUAL_BETAS = [2]
# MOMENTUM_PERCEPTUAL_BETAS = [10, 100, 1000]
# VGG_PERCEPTUAL_BETAS = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 100]
# VGG_PERCEPTUAL_BETAS = [0.01, 0.1, 0.5, 0.75, 1, 2, 5, 10, 20, 30, 60, 100]
# VGG_PERCEPTUAL_BETAS = [60, 65, 70, 75, 80, 90, 100, 110, 120, 130]
# VGG_PERCEPTUAL_BETAS = [130, 65]
VGG_PERCEPTUAL_BETAS = [740]
# VGG_PERCEPTUAL_BETAS = [680, 710, 740, 770, 800, 850 , 900]
# MSE_BETAS = [0.001, 0.005, 0.01, 0.05, 0.075, 0.1, 0.3, 0.5, 1, 2]
MSE_BETAS = [0.05]
WARMUP_BETA = 1e-1


class TrainMetrics:
    def __init__(self):
        self.recon_errors = []
        self.kl_errors = []
        self.losses = []
        self.fid = None
        self.ssim = None

    def get_file_params(self, results_directory, vae_loss_type, beta, dataset_name, suffix):
        save_path = results_directory / vae_loss_type / (dataset_name + "_beta_" + str(beta) + suffix)
        title = save_path.stem
        return save_path, title

    def create_plot(self, save_path, recon_y_label, kl_y_label, loss_y_label, title, num_of_epochs):
        fig = plt.figure(figsize=(32, 16))
        fig.suptitle(title, fontsize=40)
        x = range(1, num_of_epochs + 1)

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
            loss_trace_str += f"#epoch:{epoch + 1} ,loss:{loss:.2f}\n"
        with open(file_path, 'w') as file:
            file.write(loss_trace_str)

    def plot_results(self, results_directory, vae_loss_type, beta, recon_y_label, kl_y_label, loss_y_label,
                     dataset_name, num_of_epochs):
        # prepare plots params
        plot_save_path, plot_title = self.get_file_params(results_directory, vae_loss_type, beta, dataset_name, '.png')
        loss_trace_save_path, _ = self.get_file_params(results_directory, vae_loss_type, beta, dataset_name, '.txt')
        self.create_plot(plot_save_path, recon_y_label, kl_y_label, loss_y_label, plot_title, num_of_epochs)
        self.create_loss_trace(loss_trace_save_path)


def train(dataset_name, input_channels, train_dataloader, vae, loss_type, num_of_epochs, learning_rate, beta, device, train_results,
          perceptual_loss_network, use_pretraining_weights=False):
    if use_pretraining_weights:
        initialize_weights(vae, perceptual_loss_network, loss_type, dataset_name)

    # optimizer & scheduler
    vae_optim = torch.optim.Adam(params=vae.parameters(), lr=learning_rate)
    milestones = [num_of_epochs - 50] if NUM_EPOCHS > 100 else [150]
    vae_sched = torch.optim.lr_scheduler.MultiStepLR(vae_optim, milestones=milestones, gamma=0.1)

    # save the losses from each epoch, we might want to plot it later
    for epoch in range(num_of_epochs):
        print(f"epoch #{epoch + 1}")
        epoch_start_time = time.time()
        batch_recon_errors = []
        batch_kl_errors = []
        batch_losses = []

        for batch_i, batch in enumerate(train_dataloader):
            # print("batch #", batch_i)
            # forward pass
            x = batch[0].to(device).view(-1, input_channels, VAE.X_DIM, VAE.X_DIM)
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
            if loss_type == "momentum_perceptual" and ((batch_i + 1) % 10 == 0):
                perceptual_loss_network.update_target_weights(vae.encoder)

        train_results.recon_errors.append(np.mean(batch_recon_errors))
        train_results.kl_errors.append(np.mean(batch_kl_errors))
        train_results.losses.append(np.mean(batch_losses))
        # MultiStepLR
        vae_sched.step()

        print("epoch:{} training loss: {:.5f} epoch time: {:.3f} sec".format(epoch + 1, train_results.losses[-1],
                                                                             time.time() - epoch_start_time))

    # saving our model
    vae_fname = "/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints/" + loss_type + "/vae_" + dataset_name + "_beta_" + str(
        beta) + ".pth"
    torch.save(vae.state_dict(), vae_fname)
    print("saved checkpoint @", vae_fname)

    # momentum_encoder_fname = "./VAE_training_checkpoints/" + loss_type + "/momentum_encoder_" + dataset_name + "_beta_" + str(beta) + ".pth"
    # torch.save(perceptual_loss_network.encoder.state_dict(), momentum_encoder_fname)
    # print("saved checkpoint @", momentum_encoder_fname)


def initialize_weights(vae, perceptual_loss_network, loss_type, dataset_name):
    vae_f_name = f"/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints/starting_weights/{loss_type}/vae_mse_beta_0.1_" + dataset_name + ".pth"
    vae.load_state_dict(torch.load(vae_f_name))
    if loss_type == 'momentum_perceptual':
        momentum_encoder_f_name = f"/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints/starting_weights/{loss_type}/momentum_encoder_mse_beta_0.1_" + dataset_name + ".pth"
        perceptual_loss_network.encoder.load_state_dict(torch.load(momentum_encoder_f_name))


def get_betas_by_loss_type(vae_loss_type):
    if vae_loss_type == "mse":
        return MSE_BETAS
    elif vae_loss_type == "vgg_perceptual":
        return VGG_PERCEPTUAL_BETAS
    else:
        return MOMENTUM_PERCEPTUAL_BETAS


def get_dataset(dataset_name, is_train=None, split=None):
    root = '/home/user_115/Project/Code/Milestone1/datasets/'
    transform = torchvision.transforms.ToTensor()
    resize_transform = torchvision.transforms.Compose([transform, torchvision.transforms.Resize((VAE.X_DIM, VAE.X_DIM),
                                                                                                interpolation=torchvision.transforms.InterpolationMode.BICUBIC)])

    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root, train=is_train, transform=transform,
                                               target_transform=None, download=True)
    elif dataset_name == "svhn":
        dataset = torchvision.datasets.SVHN(root, split=split, transform=transform,
                                            target_transform=None, download=True)
    elif dataset_name == "svhn_extra":
        all_data = torchvision.datasets.SVHN(root, split="extra", transform=transform,
                                             target_transform=None, download=True)
        data_idxs = list(range(0, 140000, 2)) if is_train else list(range(1, 60001, 2))
        dataset = torch.utils.data.Subset(all_data, data_idxs)

    elif dataset_name == "flowers":
        dataset = torchvision.datasets.Flowers102(root, split=split, transform=resize_transform,
                                                  target_transform=None, download=True)
    elif dataset_name == "cars":
        dataset = torchvision.datasets.StanfordCars(root, split=split, transform=resize_transform,
                                                    target_transform=None, download=True)

    else:
        dataset = torchvision.datasets.FashionMNIST(root, train=is_train, transform=resize_transform,
                                                    target_transform=None, download=True)
    return dataset


def get_dataset_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def report_train_metrics(vae, test_dataset, train_dataloader, batch_size, train_metrics, device):
    # report fid
    num_images_for_fid_calc = len(train_dataloader.dataset) - (len(train_dataloader.dataset) % batch_size)
    fid = calc_fid_from_dataset_generate(train_dataloader, vae, batch_size, cuda=True, dims=2048, device=device,
                                         num_images=min(num_images_for_fid_calc, 50000))
    train_metrics.fid = fid
    # report ssim

    ssim = VAE_compare_input_to_recon.calc_ssim(vae, test_dataset, batch_size, device)
    train_metrics.ssim = ssim


def save_train_metrics(metrics_str, results_directory, loss_type, dataset_name, num_of_epochs):
    metrics_file_path = results_directory / f"{loss_type}" / f"{dataset_name}_fids_after_{num_of_epochs}_epochs.txt"
    with open(metrics_file_path, 'w') as mfp:
        mfp.write(metrics_str)


def main():

    # dataset_name = "cifar10"
    # dataset_name = "svhn"
    # dataset_name = "svhn_extra"
    # dataset_name = "flowers"
    # dataset_name = "stl10"
    # dataset_name = "fashion_mnist"

    results_directory = pathlib.Path("/home/user_115/Project/Results/Milestone1/VAE_training_plots/")
    weights_directory = pathlib.Path("/home/user_115/Project/Code/Milestone1/VAE_training_checkpoints")
    recon_y_label = "Reconstruction_error"
    kl_y_label = "KL_error"
    loss_y_label = "Loss"
    # here we go
    # loss_types = ["momentum_perceptual", "mse", "vgg_perceptual"]
    loss_types = ["vgg_perceptual"]
    #datasets = ["svhn_extra"]
    datasets = ["fashion_mnist"]
    perceptual_loss_network = None
    for dataset_name in datasets:
        input_channels = 3 if dataset_name != "fashion_mnist" else 1
        train_dataset = get_dataset(dataset_name, is_train=True, split="train")
        test_dataset = get_dataset(dataset_name, is_train=False, split="test")
        test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False, drop_last=True)
        samples, _ = next(iter(test_dataloader))
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        for loss_type in loss_types:
            print(f'dataset_name:{dataset_name} loss type:{loss_type}')
            if loss_type == "mse":
                BETAS = get_betas_by_loss_type(loss_type)
            elif loss_type == "vgg_perceptual":
                perceptual_loss_network = VggPerceptualLossNetwork.VggPerceptualLossNetwork(train_dataloader)
                perceptual_loss_network.eval()
                perceptual_loss_network.requires_grad_(False)
                # perceptual_loss_network = VggPerceptualLossNetworkTalVersion.VGGDistance(device=device)
                BETAS = get_betas_by_loss_type(loss_type)
            else:
                BETAS = get_betas_by_loss_type(loss_type)

            metrics_str = ""
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            for beta in BETAS:
                # check if there is gpu available, if there is, use it

                # device = torch.device("cpu")
                print("running calculations on: ", device)
                # create our model and send it to the device (cpu/gpu)
                vae = VAE.Vae(x_dim=VAE.X_DIM, in_channels=input_channels, z_dim=VAE.Z_DIM, device=device).to(device)

                if loss_type == "momentum_perceptual":
                    # sample layers for 32x32 input images
                    perceptual_loss_network = MomentumEncoder.MomentumEncoder(vae.encoder, 0.99, [0, 1, 2, 3])
                    # sample layers for 64x64 input images
                    # perceptual_loss_network = MomentumEncoder.MomentumEncoder(vae.encoder, 0.99, [0, 1, 2, 3, 4])
                    # perceptual_loss_network = MomentumEncoder.MomentumEncoder(vae.encoder, 0.99, [1, 3, 5, 7])
                    perceptual_loss_network.encoder.requires_grad_(False)
                train_metrics = TrainMetrics()
                train(dataset_name, input_channels, train_dataloader, vae, loss_type, NUM_EPOCHS, LEARNING_RATE, beta, device,
                      train_metrics, perceptual_loss_network, use_pretraining_weights=False)

                print("calculating metrics")
                vae.eval()
                test_dataset = get_dataset(dataset_name, is_train=False, split="test")
                report_train_metrics(vae, test_dataset, train_dataloader, BATCH_SIZE, train_metrics, device)
                train_metrics.plot_results(results_directory, loss_type, beta, recon_y_label, kl_y_label, loss_y_label,
                                           dataset_name, NUM_EPOCHS)
                metrics_str += f"beta:{beta}, fid:{train_metrics.fid}, ssim: {train_metrics.ssim}\n"

            save_train_metrics(metrics_str, results_directory, loss_type, dataset_name, NUM_EPOCHS)

            if NUM_EPOCHS <= 100:
                return

            print("generating new data and reconstructions")
            # plot new generated data and reconstruction
            vae = VAE.Vae(x_dim=VAE.X_DIM, in_channels=input_channels, z_dim=VAE.Z_DIM, device=device).to(device)
            vae.eval()
            VAE_gen_new_data.plot_new_generated_data(vae, dataset_name, 25, loss_type, weights_directory,
                                                     results_directory,
                                                     BETAS)
            VAE_compare_input_to_recon.plot_samples_and_recons(vae, dataset_name, input_channels, samples, 5, loss_type,
                                                               weights_directory, results_directory, BETAS, device)


if __name__ == '__main__':
    main()
