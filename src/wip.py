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
