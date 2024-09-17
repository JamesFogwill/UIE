import torch.nn as nn




import torch
import math
import torch.nn as nn


# got from colab
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # Calculate frequencies for sine and cosine functions
        embeddings = math.log(10000) / (half_dim - 1)
        # Calculate exponential of the negative frequencies
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Multiply the timestep by the negative frequencies
        embeddings = time[:, None] * embeddings[None, :]
        # Calculate the sine and cosine for each timestep in the batch
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()  # Identity mapping for skip connection

    def forward(self, x):
        identity = x  # Store the input for the skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Add the skip connection
        out = self.relu(out)  # Final ReLU
        return out



def get_index_from_list(self, vals, t, x_shape):
    """
    GOT FROM COLAB
    :param vals: a tensor of values of all time steps
    :param t: a tensor of all the time steps you want to interface with
    :param x_shape: the shape of the input images
    :return: a tensor of the values associated with the timestep tensor
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    timestep_vals = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return timestep_vals


def forward_diffusion(self, clean_batch, t, device="cpu"):
    """
    GOT FROM COLAB
    :param device: the CPU or GPU
    :param clean_batch takes batch of clean images as input
    :param t: timestep in diffusion process
    :return: the noisy version of the batch of images at timestep t

    STILL NEEDS WORK BECAUSE IT HAS TO SAMPLE AT DIFFERENT TIME STEPS
    FOR EACH IMAGE RATHER THAN THE SAME FOR THE WHOLE BATCH
    """
    # Expand dimensions of `t` to match batch size
    t = t.view(-1, 1, 1, 1)

    # generates noise in the shape of the input image
    noise = torch.randn_like(clean_batch, device=device)
    # gets the cumulative product of alphas at certain time steps
    sqrt_alpha_cumprod_t = self.get_index_from_list(self.sqrt_alpha_cumprod, t, clean_batch.shape)
    # gets the complement of the cumulative product of alphas at certain time steps
    sqrt_complement_alphas_cumprod_t = self.get_index_from_list(self.sqrt_complement_alpha_cumprod, t,
                                                                clean_batch.shape)

    # returns the noisy images and the noise
    noisy_images = sqrt_alpha_cumprod_t * clean_batch + sqrt_complement_alphas_cumprod_t * noise
    return noisy_images, noise


"""
def old_cond():
    # I should incorporate this into the upsample method rather than the downsample method
    # when I do this the loss isn't decreasing consistently which tells me this probably isnt doing anything useful
    if self.training and y0 is not None:  # Conditional input only during training
        diff_image = y0 - x
        cond_input = torch.cat([y0, diff_image], dim=1)  # Concatenate y0 and diff_image
        cond_features = self.cond_conv(cond_input)  # Process through the conditional convolution
        cond_input_2 = torch.cat([x1, cond_features], dim=1)
        x1 = self.downsample_conv(cond_input_2)

"""
"""
class Unet(nn.Module):

    def __init__(self, input_channels=3, out_channels=3, time_embedding=64, device="cuda"):
        super().__init__()

        self.device = device
        self.c_in = input_channels
        self.c_out = out_channels
        self.time_embedding = time_embedding

        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)  # takes the input and output channels
        self.sa1 = SelfAttention(128, 32)  # image resolution is halved with each down sample
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)  # takes the current channel dimension and current image resolution
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # bottleneck consisting of 3 double convs
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # decoder
        self.up1 = Up(512, 128, time_embedding)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64, time_embedding)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64, time_embedding)
        self.sa6 = SelfAttention(64, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        
        :param self:
        :param t: given random timesteps for the batch, usually of size batch_size
        :param channels: number of time dimensions which is also the columns in the vector
        :return: tensor of vector of time embeddings for each timestep where each row is the time embedding for 1 timestep
        
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)  # adds a new dimension to the end of t of type float
        t = self.pos_encoding(t, self.time_embedding)

        x1 = self.inc(x)
        # print("Shape of x1:", x1.shape)  # Add this line
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        # print("Shape of x2:", x2.shape)  # Add this line
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        # print("Shape of x3:", x3.shape)  # Add this line
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        # print("Shape of x4:", x4.shape)  # Add this line

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # extra input channels are from skip connections
        x = self.up1(x4, x3, t)
        # print("Shape of x after up1:", x.shape)  # Add this line
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        # print("Shape of x after up2:", x.shape)  # Add this line
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        # print("Shape of x after up3:", x.shape)  # Add this line
        x = self.sa6(x)
        output = self.out(x)
        return output


def generate_images2(self, model, n, inf_loader, device="cuda"):
    
    Generates new images using the trained diffusion model.

    Args:
        model: The trained U-Net model.
        n: The number of images to generate.
        inf_loader: inference image data loader
        device: the device where image processing happens

    Returns:
        A tensor of generated images (shape: (n, 3, img_size, img_size)).
    
    logging.info(f"Sampling {n} new images....")  # Log a message indicating the start of sampling

    model.eval()  # Set the model to evaluation mode (disables dropout, batch norm, etc.)

    generated_images_list = []
    input_images_list = []

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
                input_images_list.append(input_image)

    model.train()

    # Post-process all generated images together
    all_generated_images = torch.cat(generated_images_list, dim=0)
    all_generated_images = (all_generated_images.clamp(-1, 1) + 1) / 2
    all_generated_images = (all_generated_images * 255).type(torch.uint8)

    return all_generated_images

    def generate_images(self, model, n, inf_loader, device="cuda"):
        
        Generates new images using the trained diffusion model.

        Args:
            model: The trained U-Net model.
            n: The number of images to generate.
            inf_loader: inference image data loader
            device: the device where image processing happens

        Returns:
            A tensor of generated images (shape: (n, 3, img_size, img_size)).
        
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
                predicted_noise = model(x, t)

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


Testing of the model

# Load the state dictionary from the specified checkpoint
checkpoint_path = os.path.join("../", "results", "models", "checkpoints", "ckpt_epoch_25.pt")
model.load_state_dict(torch.load(checkpoint_path))

# Set the model to evaluation mode for testing
model.eval()

# Generate images using the test loader
with torch.no_grad():
    # generate 3 images for each gt image in the test set
    sampled_images = noise_scheduler.generate_images_metrics(model, n=1, inf_loader=test_loader)

    # write these images to the tensor board
    sampled_images_grid = torchvision.utils.make_grid(sampled_images)
    writer.add_image(f'Test Sample Images', sampled_images_grid)

# Save or visualize the generated images
DatasetUtils.save_images(images=sampled_images, path=os.path.join("../", "results", "test_images", "generated.jpg"))




class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        
        
        
        
        class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.channel_attention = ChannelAttention(256)

    def forward(self, rgb_features, hsv_features):
        # concatenate the feature maps together
        fused_features = torch.cat([rgb_features, hsv_features], dim=1)
        # conv layer to extrac relationships between the feature maps
        fused_features = self.conv(fused_features)
        # channel attention to select more relevant channels of H,S or V
        return self.channel_attention(fused_features)
"""