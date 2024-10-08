# Beta Variational Autoencoder (β-VAE) for Image Generation

This repository contains a Jupyter notebook implementing a Beta Variational Autoencoder (β-VAE) for image generation. The β-VAE is an extension of the Vanilla VAE that introduces a hyperparameter β to control the trade-off between reconstruction quality and disentanglement in the latent space.

## Contents

- `08-10-2024 Beta VAE.ipynb`: The main Jupyter notebook containing the implementation of the β-VAE.

## Features

- Implementation of a β-VAE using PyTorch
- Training on a dataset of images
- Generation of new images from the learned latent space
- Visualization of training progress and generated images
- Exploration of latent space disentanglement

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

## Usage

1. Open the `08-10-2024 Beta VAE.ipynb` notebook in a Jupyter environment.
2. Run the cells in order to train the β-VAE and generate images.
3. The notebook will save generated images in the `generated_images` folder and model checkpoints in the `saved_models` folder.

## Model Architecture

The β-VAE architecture is similar to the Vanilla VAE but with modifications to the loss function:

### Encoder
- Convolutional layers to extract features from input images
- Fully connected layers to map features to mean and log-variance of the latent space distribution

### Latent Space
- Reparameterization trick to sample from the latent space distribution

### Decoder
- Fully connected layers to map latent space samples to feature maps
- Transposed convolutional layers to reconstruct the image from feature maps

## Training Process

The β-VAE is trained using a modified loss function:
1. Reconstruction loss: Measures how well the decoder can reconstruct the input image
2. KL divergence loss: Ensures the latent space distribution is close to a standard normal distribution
3. β parameter: Balances the importance of the KL divergence term

The total loss is: Reconstruction Loss + β * KL Divergence Loss

The model is optimized using the Adam optimizer.

## Results

The notebook demonstrates:
- Training progress with loss curves
- Original images vs. reconstructed images
- Generated images from random latent space samples
- Interpolation between images in the latent space
- Analysis of latent space disentanglement

## Future Improvements

- Experiment with different β values to observe the impact on disentanglement
- Implement advanced techniques like cyclical annealing of β
- Apply the model to different datasets and compare disentanglement results

## References

- Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR.
- PyTorch documentation: https://pytorch.org/docs/stable/index.html

## License

This project is open-source and available under the MIT License.
