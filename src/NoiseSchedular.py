import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, number_of_sampled_timesteps):
        return torch.randint(low=1, high=self.time_steps, size=(number_of_sampled_timesteps,))

    def generate_images(self, model, n, inf_loader, device="cuda"):
        """
        Generates new images using the trained diffusion model.

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

        with torch.no_grad():  # Disable gradient calculation during inference to save memory
            # Initialize with pure noise
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

            # Iterate over time-steps in reverse order (from T-1 down to 1)
            for i in tqdm(reversed(range(1, self.time_steps)), position=0):  # tqdm for progress bar
                # Create a tensor of the current timestep for each image in the batch
                t = (torch.ones(n) * i).long().to(self.device)

                # Predict the noise at the current timestep
                predicted_noise = model(x, t, )

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

        model.train()  # Set the model back to training mode after sampling

        # Post-process the generated images
        x = (x.clamp(-1, 1) + 1) / 2  # Scale the pixel values to the [0, 1] range
        x = (x * 255).type(torch.uint8)  # Convert to 8-bit unsigned integers for saving

        return x  # Return the generated images

# To do when conditional input works
# each 10 epochs I should test the model on the inference data during the training. After the final epoch of training
# I should have another set of images that evaluate the model completely unseen

    def generate_images2(self, model, n, inf_loader, device="cuda"):
        """
        Generates new images using the trained diffusion model.

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

        generated_images_list = []

        with torch.no_grad():  # Disable gradient calculation during inference to save memory

            for batch in inf_loader:
                input_images, _, _ = batch
                input_images = input_images.to(device)
                # Generate 'n' images conditioned on each ground truth image in the batch
                for input_image in input_images:
                    x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
                    for i in tqdm(reversed(range(1, self.time_steps)), position=0):
                        t = (torch.ones(n) * i).long().to(self.device)
                        predicted_noise = model(x, t, y0=input_image.unsqueeze(0))  # Condition on gt_image

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

        model.train()

        # Post-process all generated images together
        all_generated_images = torch.cat(generated_images_list, dim=0)
        all_generated_images = (all_generated_images.clamp(-1, 1) + 1) / 2
        all_generated_images = (all_generated_images * 255).type(torch.uint8)
        return all_generated_images


