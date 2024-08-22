import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def display_image_pair(input_images, gt_images):
    """
    Displays the first image from the input_images and gt_images batches.

    Args:
        input_images: A batch of input images (tensor).
        gt_images: A batch of ground truth images (tensor).
    """

    # Extract the first image from each batch
    first_input_image = input_images[0].cpu().detach()
    first_gt_image = gt_images[0].cpu().detach()

    # Denormalize images if necessary (assuming your transformations normalized them)
    first_input_image = (first_input_image + 1) / 2
    first_gt_image = (first_gt_image + 1) / 2

    # Display the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(first_input_image.permute(1, 2, 0))  # Convert to HWC for matplotlib
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(first_gt_image.permute(1, 2, 0))  # Convert to HWC for matplotlib
    plt.title("Ground Truth Image")
    plt.axis('off')

    plt.show()


def cpdm_loss(model, x0, y0, noise_scheduler, mse_loss):
    """
    Calculates the CPDM loss as described in the paper.

    Args:
        model: The U-Net model.
        x0: Batch of original (degraded) images.
        y0: Batch of corresponding ground truth (clean) images.
        noise_scheduler: The noise scheduler used for the diffusion process.
        mse_loss: An instance of the MSE loss function.

    Returns:
        The calculated CPDM loss.
    """

    # Sample timesteps
    t = noise_scheduler.sample_timesteps(x0.shape[0]).to(x0.device)

    # Forward diffusion process
    xt, noise = noise_scheduler.noise_images(x0, t)

    # Get model prediction
    predicted_noise = model(xt, t, y0=y0)

    # Calculate the loss
    loss = mse_loss(noise, predicted_noise)
    return loss
