<h1 align="center">
  <br>
BYOGM Project
  <br>
</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/tal-rubinstein-131450a6
">Tal Rubinstein</a> 
    <a href="https://www.linkedin.com/in/ron-ben-zalen-418525174">Ron Ben Zalen</a>
  </p>
  <p align="center">
    <a href="https://github.com/taldatech">Supervised by Tal Daniel</a>
  </p>
<h4 align="center">Official repository of the project</h4>

<p align="center">
  <img src="https://github.com/ronbenz/BYOGM-Project/blob/master/Results/VAE_compare_input_to_recon/momentum_perceptual/fashion_mnist_compare_input_to_recon.png" height="120">
  <img src="https://github.com/ronbenz/BYOGM-Project/blob/master/Results/VAE_new_generated_data/momentum_perceptual/fashion_mnist_new_generated_data.png" height="120">

</p>
<p align="center">
  <img src="https://github.com/ronbenz/BYOGM-Project/blob/master/Results/VAE_compare_input_to_recon/momentum_perceptual/svhn_extra_compare_input_to_recon.png" height="120">
  <img src="https://github.com/ronbenz/BYOGM-Project/blob/master/Results/VAE_new_generated_data/momentum_perceptual/svhn_extra_new_generated_data.png" height="120">
</p>

# Momentum VAE


> **Abstract:** *Training generative models using per-pixel loss often results in poor generated images.
Better perceptual quality can be achieved using feature-based loss functions which use pre-trained models as feature extractors. Unfortunately, pre-trained models usually come with some drawbacks. First, they are often pre-trained on a task that is unrelated to the main task. In addition, they are usually big and computationally heavy. Our project goal was to Investigate the need for pre-trained networks during VAE training by replacing them with a bootstrapped version of the trained model. As part of the project, the standard MSE loss which is used in the process of VAE training was replaced by a Perceptual loss based on features extracted from an EMA bootstrapped version of the VAE encoder. The code for the project was mainly written using Pytorch and the trained model was evaluated on Fashion MNIST and SVHN datasets.


## Repository Organization

|File name         | Content |
|----------------------|------|
|`/Code`| Directory containing all of the project's code|
|`/Results`| Directory containing reconstruction and generation results on Fashion MNIST and SVHN|


## Credits
* Bootstrap your own latent: A new approach to self-supervised Learning, Jean-Bastien Grill et al., - [Code](https://github.com/lucidrains/byol-pytorch), [Paper](https://arxiv.org/abs/2006.07733).
* FID is calculated natively in PyTorch using Seitzer implementation - [Code](https://github.com/mseitzer/pytorch-fid)



