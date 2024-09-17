import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import logging
from PIL import Image
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from skimage.exposure import rescale_intensity


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


def inference(model, scheduler, conditonal_images, T, epoch, testing=False,device="cuda"):
    """
    :param testing:
    :param epoch:
    :param conditonal_images: the conditional image batch
    :param device: device where image generation happens
    :param model: model to use to generate images
    :param scheduler: noise scheduler
    :param T: number of timesteps
    :return: final denoised image
    """
    if testing is False:
        checkpoint_path = os.path.join("../", "results", "models", "checkpoints", f"ckpt_epoch_{epoch+1}.pt")
        model.load_state_dict(torch.load(checkpoint_path))

    model.eval()

    batch_size = conditonal_images.shape[0]
    print(batch_size)
    # Sample initial noise and ensure y0 is on the correct device
    x = torch.randn((batch_size, 3, 64, 64)).to(device)
    y0 = conditonal_images.to(device)
    with torch.no_grad():

        for i in tqdm(reversed(range(1, T)), position=0):

            t = (torch.ones(batch_size) * i).long().to(device)
            # x_cond = torch.cat([x, y0], dim=1)
            predicted_noise = model(x, t, y0=y0)

            alpha_t, alpha_cump_t, beta_t = scheduler.get_alphas(t)

            # Retrieve alpha, alpha_hat, and beta values for the current timestep
            alpha = alpha_t[:, None, None, None]
            alpha_hat = alpha_cump_t[:, None, None, None]
            beta = beta_t[:, None, None, None]

            # Sample new noise if not at the last timestep
            if i > 1:
                noise = torch.randn_like(x)
            else:  # No noise at the last step to get the final clean image
                noise = torch.zeros_like(x)

            # Reverse diffusion step: calculate the less noisy image 'x'
            x = 1 / torch.sqrt(alpha) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                beta) * noise

    model.train()
    # 6. Return the final denoised image (x0)
    return x


def calculate_metrics(generated_images, gt_images):
    """
  Calculates average PSNR, SSIM, and MSE for a batch of generated images.

  Args:
      generated_images: A PyTorch tensor of generated images (shape: [batch_size, channels, height, width]).
      gt_images: A PyTorch tensor of ground truth images (same shape as generated_images).

  Returns:
      avg_psnr: Average Peak Signal-to-Noise Ratio.
      avg_ssim: Average Structural Similarity Index Measure.
      avg_mse: Average Mean Squared Error.
  """

    psnr_values = []
    ssim_values = []
    mse_values = []

    # Convert to numpy and compute metrics
    generated_images_np = generated_images.cpu().numpy().transpose(0, 2, 3, 1)
    gt_images_np = gt_images.cpu().numpy().transpose(0, 2, 3, 1)

    for gen_img, gt_img in zip(generated_images_np, gt_images_np):
        # Convert to grayscale
        gen_img_gray = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)
        gt_img_gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)

        # normalise for MSE metric
        gen_img_normalized = rescale_intensity(gen_img, in_range=(gen_img.min(), gen_img.max()), out_range=(0, 1))
        gt_img_normalized = rescale_intensity(gt_img, in_range=(gen_img.min(), gen_img.max()), out_range=(0, 1))

        # Calculate metrics
        ssim = structural_similarity(gt_img_gray, gen_img_gray, data_range=gt_img_gray.max() - gt_img_gray.min())
        psnr = peak_signal_noise_ratio(gt_img, gen_img, data_range=255)
        mse = mean_squared_error(gt_img_normalized, gen_img_normalized)

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        mse_values.append(mse)

    # Calculate average metrics
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_mse = sum(mse_values) / len(mse_values)

    return avg_psnr, avg_ssim, avg_mse


def save_images(images, path):
    if images.dtype != torch.uint8:
        images = (images.clamp(0, 1) * 255).to(torch.uint8)  # Normalize and convert to uint8 if needed

    grid = torchvision.utils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def tensor_to_tensorboard_image(sampled_images):
    """Converts a tensor of sampled images to a format suitable for TensorBoard.

    Args:
        sampled_images: A PyTorch tensor of generated images (shape: [batch_size, channels, height, width]).
                      Assumes values are in the range [-1, 1].

    Returns:
        A PyTorch tensor ready to be added to TensorBoard using `add_image`.
    """

    # Clamp values to [-1, 1] and rescale to [0, 1]
    sampled_images = (sampled_images.clamp(-1, 1) + 1) / 2

    # Convert to uint8 for image representation
    sampled_images = (sampled_images * 255).type(torch.uint8)

    # Create a grid for visualization
    grid = torchvision.utils.make_grid(sampled_images)

    return grid, sampled_images


def create_paired_grid(grid_1, grid_2, grid_3=None):
    """Combines two image grids into a paired dataset grid.

  Args:
      grid_1: A PyTorch tensor representing the first image grid.
      grid_2: A PyTorch tensor representing the second image grid.
            Both grids should have the same number of images.

  Returns:
      A PyTorch tensor representing the combined paired grid.
  """
    if grid_3 is None:
        # Create separate grids for input and generated images
        grid_1 = torchvision.utils.make_grid(grid_1, nrow=grid_1.shape[0])
        grid_2 = torchvision.utils.make_grid(grid_2, nrow=grid_2.shape[0])

        # Stack the grids vertically
        final_grid = torch.cat([grid_1, grid_2],
                            dim=1)  # Concatenate along the height dimension (dim=1)
    else:
        grid_1 = torchvision.utils.make_grid(grid_1, nrow=grid_1.shape[0])
        grid_2 = torchvision.utils.make_grid(grid_2, nrow=grid_2.shape[0])
        grid_3 = torchvision.utils.make_grid(grid_3, nrow=grid_3.shape[0])

        final_grid = torch.cat([grid_1, grid_2, grid_3], dim=1)

    return final_grid


def inference_full(inference_loader, model, noise_scheduler, time_steps, writer, epoch, testing=False):

    all_sampled_images = []  # Accumulate sampled images across batches
    all_gt_images = []  # Accumulate ground truth images across batches
    all_input_images = []

    _testing = False

    if testing is True:
        _testing = True

    # run inference
    for batch_idx, batch in enumerate(inference_loader):
        # usually false
        if testing is False:
            if batch_idx == 5:
                break

        in_imgs, gt_imgs = batch

        for i in range(in_imgs.shape[0]):
            in_img = in_imgs[i].unsqueeze(0)
            gt_img = gt_imgs[i].unsqueeze(0)
            sampled_img = inference(model=model, scheduler=noise_scheduler, conditonal_images=in_img, T=time_steps, epoch=epoch, testing=_testing)

            all_sampled_images.append(sampled_img)
            all_gt_images.append(gt_img)
            all_input_images.append(in_img)

    # Concatenate all sampled and ground truth images into single tensors
    all_sampled_images = torch.cat(all_sampled_images, dim=0)
    all_gt_images = torch.cat(all_gt_images, dim=0)
    all_input_images = torch.cat(all_input_images, dim=0)

    all_gt_images = all_gt_images.to("cpu")
    all_sampled_images = all_sampled_images.to("cpu")
    all_input_images = all_input_images.to("cpu")

    # make grids and convert to suitable formate for display
    gt_image_grid, gt_imgs_ = tensor_to_tensorboard_image(all_gt_images)
    sampled_image_grid, sampled_imgs_ = tensor_to_tensorboard_image(all_sampled_images)
    input_image_grid, _ = tensor_to_tensorboard_image(all_input_images)

    # Combine the grids
    paired_grid = create_paired_grid(gt_image_grid, sampled_image_grid, grid_3= input_image_grid)

    if testing is False:
        writer.add_image(f'Paired Images for Epoch: {epoch + 1}', paired_grid)
        save_images(images=all_sampled_images,
                    path=os.path.join("../", "results", "sample_images", f"{epoch}.jpg"))
    else:
        writer.add_image(f'Paired Test images', paired_grid)
        save_images(images=paired_grid,
                    path=os.path.join("../", "results", "test_images", "generated.jpg"))

    # calculate metrics
    PSNR, SSIM, MSE = calculate_metrics(sampled_imgs_, gt_imgs_)
    if testing is False:
        # Add metrics to TensorBoard
        writer.add_scalar('PSNR', PSNR, epoch)
        writer.add_scalar('SSIM', SSIM, epoch)
        writer.add_scalar('MSE', MSE, epoch)
    else:
        # Add metrics to TensorBoard
        writer.add_scalar('PSNR', PSNR)
        writer.add_scalar('SSIM', SSIM)
        writer.add_scalar('MSE', MSE)

    # add images to tensor board and save them and the state dictionary
    sampled_img_grid = torchvision.utils.make_grid(all_sampled_images)
    # writer.add_image(f'Epoch[{epoch + 1}] Sampled images', sampled_img_grid)

