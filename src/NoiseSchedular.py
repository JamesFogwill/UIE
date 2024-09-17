import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.exposure import rescale_intensity

import logging
from tqdm import tqdm
from torch import optim

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class NoiseSchedular(nn.Module):
    """
    A NoiseSchedular for a diffusion model generating accurate underwater images.
    parameters: img_size = size of the image e.g. 64x64, time_step = number of diffusion steps,
    beta_start = starting value for noise variance default 1e-4
    beta_end = ending value for noise variance 0.02
    """

    def __init__(self, img_size=64, time_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__()

        """
        initialises network layers and modules
        inherits from parent class via super() - can use methods from nn.Module
        sets up requirements for the forward and backward diffusion process
        
        self.beta = linear schedule, increasing from min to max, represents portion of noise remaining at each time step
        self.alpha = alphas represent portion of original signal remaining at each time step
        self.alpha_cumprod = # cumulative product of alphas, how much of original signal remains at each time step, 
        α_1 * α_2 * ... * α_t
        self.sqrt_alpha_cumprod = calculates square root of cumprod used in forward process
        self.sqrt_complement_alpha_cumprod =
        
        self.alpha_cumprod_prev = takes the alpha_cumprod tensor, removes the last index, pads the beginning with 1.0 
        and nothing on the end 
        self.sqrt_recip_alphas = calcs sqrt of the reciprocals of the alphas
        self.posterior_variance = Used in reverse diffusion process to estimate how much noise should be removed
        """

        self.img_size = img_size
        self.time_steps = time_steps
        self.device = device
        self.beta_start = beta_start
        self.beta_end = beta_end

        # used in forward diffusion process
        self.beta = torch.linspace(beta_start, beta_end, time_steps)
        self.alpha = 1. - self.beta
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)

    def noise_images(self, x, t):
        # sqrt and reshape tensor for broadcasting
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sigma = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sigma * noise, noise

    def get_alphas(self, t):
        alpha_t = self.alpha[t]
        alpha_cump_t = self.alpha_hat[t]
        beta_t = self.beta[t]

        return alpha_t, alpha_cump_t, beta_t

    def sample_timesteps(self, number_of_sampled_timesteps):
        return torch.randint(low=1, high=self.time_steps, size=(number_of_sampled_timesteps,))

    def generate_images_metrics(self, model, batch_size, loader, device="cuda"):
        """
        Generates new images using the trained diffusion model.
        takes a random image from each batch to recreate to save time.
        pairs the image with the generated image

        Args:
            model: The trained U-Net model.
            n: The number of images to generate.
            loader: inference image data loader
            device: the device where image processing happens

        Returns:
            A tensor of generated images (shape: (n, 3, img_size, img_size)).
        """
        model.eval()  # Set the model to evaluation mode (disables dropout, batch norm, etc.)

        # image lists
        generated_images_list = []
        input_images_list = []
        gt_images_list = []

        with torch.no_grad():  # Disable gradient calculation during inference to save memory

            for batch in loader:
                # Get a single image from the batch
                input_images, gt_images = batch
                input_images = input_images.to(device)  # underwater image

                # Generate 'n' images conditioned on the randomly selected input image
                x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(self.device)

                for i in tqdm(reversed(range(1, self.time_steps)), position=0):

                    t = self.sample_timesteps(batch_size).to(device)

                    # predicted_noise = model(x, t)

                    predicted_noise = model(x, t, y0=input_images)

                    # Retrieve alpha, alpha_hat, and beta values for the current timestep
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]

                    # Sample new noise if not at the last timestep
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:  # No noise at the last step to get the final clean image
                        noise = torch.zeros_like(x)

                    # Reverse diffusion step: calculate the less noisy image 'x'
                    x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                        beta) * noise

                generated_images_list.append(x)
                input_images_list.append(input_images)
                gt_images_list.append(gt_images)

        model.train()

        # Post-process all generated images together
        all_generated_images = torch.cat(generated_images_list, dim=0)
        all_generated_images = (all_generated_images.clamp(-1, 1) + 1) / 2
        all_generated_images = (all_generated_images * 255).type(torch.uint8)

        all_input_images = torch.cat(input_images_list, dim=0)
        all_input_images = (all_input_images.clamp(-1, 1) + 1) / 2
        all_input_images = (all_input_images * 255).type(torch.uint8)

        all_gt_images = torch.cat(gt_images_list, dim=0)
        all_gt_images = (all_gt_images.clamp(-1, 1) + 1) / 2
        all_gt_images = (all_gt_images * 255).type(torch.uint8)
        print(all_gt_images.shape)

        # Calculate evaluation metrics
        psnr_values = []
        ssim_values = []
        mse_values = []

        # Convert to numpy and compute metrics
        generated_images_np = all_generated_images.cpu().numpy().transpose(0, 2, 3, 1)
        gt_images_np = all_gt_images.cpu().numpy().transpose(0, 2, 3, 1)

        for gen_img, gt_img in zip(generated_images_np, gt_images_np):
            print(f"Generated image min/max values: {gen_img.min()}, {gen_img.max()}")
            print(f"Ground truth image min/max values: {gt_img.min()}, {gt_img.max()}")

            # convert to grey scale
            gen_img_gray = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)
            gt_img_gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)

            # normalise for MSE metric
            gen_img_normalized = rescale_intensity(gen_img, in_range=(gen_img.min(), gen_img.max()), out_range=(0, 1))
            gt_img_normalized = rescale_intensity(gt_img, in_range=(gen_img.min(), gen_img.max()), out_range=(0, 1))

            # calculate metrics
            ssim = structural_similarity(gt_img_gray, gen_img_gray, data_range=255)
            psnr = peak_signal_noise_ratio(gt_img, gen_img, data_range=255)
            mse = mean_squared_error(gt_img_normalized, gen_img_normalized)

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            mse_values.append(mse)

        # Calculate average metrics
        avg_psnr = sum(psnr_values) / len(psnr_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        avg_mse = sum(mse_values) / len(mse_values)

        # Create separate grids for input and generated images
        input_grid = torchvision.utils.make_grid(all_input_images, nrow=all_input_images.shape[0]).to("cpu")
        generated_grid = torchvision.utils.make_grid(all_generated_images, nrow=all_generated_images.shape[0]).to("cpu")
        gt_grid = torchvision.utils.make_grid(all_gt_images, nrow=all_gt_images.shape[0]).to("cpu")

        # Stack the grids vertically
        final_grid = torch.cat([input_grid, generated_grid, gt_grid],
                               dim=1)  # Concatenate along the height dimension (dim=1)

        return final_grid, avg_psnr, avg_ssim, avg_mse

    def generate_images(self, model, n, inf_loader, device="cuda"):
        """
        Generates new images using the trained diffusion model.
        takes a random image from each batch to recreate to save time.
        pairs the image with the generated image

        Args:
            model: The trained U-Net model.
            n: The number of images to generate.
            inf_loader: inference image data loader
            device: the device where image processing happens

        Returns:
            A tensor of generated images (shape: (n, 3, img_size, img_size)).
        """
        logging.info(f"Sampling {n} new images....")  # Log a message indicating the start of sampling

        model.eval()  # Set the model to evaluation mode (disables dropout, batch norm, etc.)

        # image lists
        generated_images_list = []
        input_images_list = []
        gt_images_list = []

        with torch.no_grad():  # Disable gradient calculation during inference to save memory

            for batch in inf_loader:
                # Get a single image from the batch
                input_images, gt_images, _, input_hsv = batch
                input_images = input_images.to(device)
                input_hsv = input_hsv.to(device)
                gt_images = gt_images.to(device)

                # Randomly select one image from the batch
                random_index = torch.randint(0, input_images.shape[0], (1,)).item()
                input_image = input_images[random_index]
                input_image_hsv = input_hsv[random_index]
                gt_image = gt_images[random_index]

                # Generate 'n' images conditioned on the randomly selected input image
                x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

                for i in tqdm(reversed(range(1, self.time_steps)), position=0):
                    t = (torch.ones(n) * i).long().to(self.device)

                    predicted_noise = model(x, t, y0=input_image.unsqueeze(0))
                    # predicted_noise = model(x, t)

                    # Retrieve alpha, alpha_hat, and beta values for the current timestep
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]

                    # Sample new noise if not at the last timestep
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:  # No noise at the last step to get the final clean image
                        noise = torch.zeros_like(x)

                    # Reverse diffusion step: calculate the less noisy image 'x'
                    x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                        beta) * noise

                generated_images_list.append(x)
                input_images_list.append(input_image.unsqueeze(0))
                gt_images_list.append(gt_image.unsqueeze(0))

        model.train()

        # Post-process all generated images together
        all_generated_images = torch.cat(generated_images_list, dim=0)
        all_generated_images = (all_generated_images.clamp(-1, 1) + 1) / 2
        all_generated_images = (all_generated_images * 255).type(torch.uint8)

        all_input_images = torch.cat(input_images_list, dim=0)
        all_input_images = (all_input_images.clamp(-1, 1) + 1) / 2
        all_input_images = (all_input_images * 255).type(torch.uint8)

        all_gt_images = torch.cat(gt_images_list, dim=0)
        all_gt_images = (all_gt_images.clamp(-1, 1) + 1) / 2
        all_gt_images = (all_gt_images * 255).type(torch.uint8)

        # Create separate grids for input and generated images
        input_grid = torchvision.utils.make_grid(all_input_images, nrow=all_input_images.shape[0])
        generated_grid = torchvision.utils.make_grid(all_generated_images, nrow=all_generated_images.shape[0])
        gt_grid = torchvision.utils.make_grid(all_gt_images, nrow=all_gt_images.shape[0])

        # Stack the grids vertically
        final_grid = torch.cat([input_grid, generated_grid, gt_grid],
                               dim=1)  # Concatenate along the height dimension (dim=1)

        return final_grid
