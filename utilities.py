import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import main
from main import *


def psnr(real_image, fake_image):
    """
    This function computes the PSNR value between the real images and the fake image.
    """
    # Compute the MSE between real and fake images
    mse_loss = torch.mean((real_image - fake_image) ** 2)

    # Compute PSNR value using MSE
    max_intensity = 1.0  # Assuming pixel values are in the range [0, 1]
    psnr_value = 20 * torch.log10(max_intensity / torch.sqrt(mse_loss))

    # Extract PSNR value as a scalar
    return psnr_value.item()



def convert_lab_to_rgb(L, ab):
    """
    Converting a batch of images from L*a*b to RGB.
    """
    rgb_imgs = []

    # De-normalizing the L*a*b values.
    L = (L + 1) * 50
    ab *= 110

    # Concatenate L and ab channels along the channel dimension
    Lab = torch.cat([L, ab], dim=1)

    # Transpose dimensions to match expected input for lab2rgb function
    Lab_permuted = Lab.permute(0, 2, 3, 1).cpu().numpy()

    # Converting each image in the batch to RGB
    for img in Lab_permuted:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)

    # Stack the RGB images along the batch dimension
    rgb_imgs = np.stack(rgb_imgs, axis=0)

    return rgb_imgs



def calc_gradient_penalty(real_images, fake_images):
    """
    Calculates the gradient penalty loss for WGAN with gradient penalty.
    """

    # Random weight term for interpolation between real and fake images
    alpha = torch.rand((real_images.size(0), 1, 1, 1), device='cuda')

    # Get random interpolation between real and fake images
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images))
    interpolates = interpolates.requires_grad_(True)

    # Calculate critic scores for the interpolated images
    pred = critic(interpolates)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        inputs=interpolates,
        outputs=pred,
        grad_outputs=torch.ones(pred.size(), device='cuda'),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten gradients
    gradients = gradients.view(gradients.size(0), -1)

    # Compute gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)

    return gradient_penalty.mean()


def plot_images(greyscale_images, fake_colorized_images, orig_color_images, epoch=1, batch=0, num_images=8, image_dir_path=None, mode="training"):
    """
    This function plots or saves the images.
    In order plot, set image_dir_path=None, else specify a path.
    """
    file_name = os.path.join(image_dir_path, f"{mode}_batch_{batch}" if mode=='test' else f"{mode}_epoch_{epoch+1}")

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        # Plot greyscale image - The 'L' component of the image.
        plt.subplot(3, num_images, i + 1)
        plt.imshow(greyscale_images[i].cpu().squeeze(), cmap='gray')
        if i==0:
            plt.title('Greyscale')
        plt.axis('off')

        # Plot colorized image
        plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(torch.from_numpy(fake_colorized_images[i]).cpu())
        if i==0:
            plt.title('Colorized')
        plt.axis('off')

        # Plot original color image
        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(torch.from_numpy(orig_color_images[i]).cpu())
        if i==0:
            plt.title('Original Color')
        plt.axis('off')

    # This section saves the images or plots them
    if image_dir_path==None or main.test==True:
        plt.show()
    else:
        plt.savefig(file_name + '.png', format='png', bbox_inches='tight', dpi=300)  # Save the figure as an image
        plt.close()


# def plot_loss(loss_1, title_1, loss_2, title_2, mode):
#     fig, axs = plt.subplots(1, 2, figsize=(15,5))
#     fig.suptitle(f"{mode}")

#     axs[0].plot(loss_1, color='green', label=title_1)
#     axs[0].set_xlabel("Global Batch Steps")
#     axs[0].set_ylabel("Loss")
#     axs[0].set_title(title_1)
#     axs[0].legend()

#     axs[1].plot(loss_2, color='blue', label=title_2)
#     axs[1].set_xlabel("Global Batch Steps")
#     axs[1].set_ylabel("Loss")
#     axs[1].set_title(title_2)
#     axs[1].legend()

#     plt.show()


def plot_loss(loss_1, title_1, loss_2, title_2, loss_3=None, title_3=None, mode=None):

    num_of_graphs = 2
    num_of_graphs += 1 if loss_3 is not None else 0

    fig, axs = plt.subplots(1, num_of_graphs, figsize=(15,5))
    fig.suptitle(f"{mode}")

    axs[0].plot(loss_1, color='green', label=title_1)
    axs[0].set_xlabel("Global Batch Steps")
    axs[0].set_ylabel("Loss")
    axs[0].set_title(title_1)
    axs[0].legend()

    axs[1].plot(loss_2, color='blue', label=title_2)
    axs[1].set_xlabel("Global Batch Steps")
    axs[1].set_ylabel("Loss")
    axs[1].set_title(title_2)
    axs[1].legend()

    if loss_3:
        axs[2].plot(loss_3, color='red', label=title_3)
        axs[2].set_xlabel("Global Batch Steps")
        axs[2].set_ylabel("Loss")
        axs[2].set_title(title_3)
        axs[2].legend()

    plt.show()


def plot_psnr(psnr_values, mode):
    # Plotting the PSNR values
    plt.figure(figsize=(10, 5))
    plt.plot(psnr_values)
    plt.xlabel('Global Batch Steps')
    plt.ylabel('PSNR in dB')
    plt.title(f'PSNR of {mode}')
    plt.show()


def plot_loss_gp(crit_loss, title):
    plt.figure(figsize=(10, 5))
    plt.plot(crit_loss)
    plt.xlabel('Global Batch Steps')
    plt.ylabel('Loss')
    plt.title(title)
    plt.ylim(bottom=-5, top=10)
    plt.show()


