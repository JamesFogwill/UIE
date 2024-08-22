import torch
import torch.nn as nn
import torch.nn.functional as F
import diff_modules as dm
import pdb


# We make a class for each block for re-usability in other things we do

# ALL THIS IS TAKEN FROM https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))  # add the original image to the image after convolutions and use
            # activation function to get the residual information
        else:
            return self.double_conv(x)


class SKC(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=8):
        super().__init__()

        # 'Fuse' Part of SKC
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                              padding=1)  # Convolution on concatenated features
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Conv2d(in_channels * 2, in_channels * 2 // reduction_ratio, kernel_size=1)  # Channel reduction
        self.fc2 = nn.Conv2d(in_channels * 2 // reduction_ratio, in_channels * 2, kernel_size=1)  # Channel expansion

        # Two parallel 1x1 convolutions to generate feature descriptors
        self.conv_v1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv_v2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, RGB, HSV):
        # Fuse features from RGB and HSV
        L = torch.cat([RGB, HSV], dim=1)  # Concatenate along channel dimension

        # Convolution and Global Average Pooling
        conv_out = self.conv(L)
        s = self.gap(conv_out)  # Channel-wise statistics

        # Compact feature representation
        z = self.fc1(s)
        z = F.relu(z)
        z = self.fc2(z)

        # Generate feature descriptors
        v1 = self.conv_v1(z)
        v2 = self.conv_v2(z)

        # 'Select' Part of SKC
        # Calculate attention activations (similar to self-attention)
        attention_weights = F.softmax(torch.cat([v1, v2], dim=1), dim=1)
        s1, s2 = attention_weights.chunk(2, dim=1)

        # Select features from RGB and HSV based on attention
        VRGB = s1 * RGB
        VHsv = s2 * HSV

        # Concatenate the selected features
        U = torch.cat([VRGB, VHsv], dim=1)
        return U


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim=64):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),  # max pool to half the dimensions, 2 means half all dims
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels)
        )

        self.time_embedding_layer = nn.Sequential(
            nn.SiLU(),  # sigmoid linear unit activation linear layer that transforms time embedding vector into
            # vector of correct size enabling time information to be added to the feature map
            nn.Linear(
                time_embedding_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.down_sample(x)
        # adds 2 dimensions and then makes them = to the height(x.shape[-2]) and width(x.shape[-1]) of the image
        embedding = self.time_embedding_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        # now that the time embedding is the same dimensions we can add them together
        return x + embedding


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding=64):
        super().__init__()

        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.time_embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding, out_channels),
        )

    def forward(self, x, x_skip, t):
        x = self.up_sample(x)
        x = torch.cat([x, x_skip], dim=1)  # this is where the skip connection is added
        x = self.conv(x)
        embedding = self.time_embedding_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + embedding


# Complete copy from the github
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


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling

        # Bottleneck MLP to learn channel-wise dependencies
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

        self.sigmoid = nn.Sigmoid()  # To produce attention weights between 0 and 1

    def forward(self, x):
        # Get global average and max pooled features
        avg_pool = self.avg_pool(x).squeeze(-1).squeeze(-1)  # Shape: (batch_size, channels)
        max_pool = self.max_pool(x).squeeze(-1).squeeze(-1)  # Shape: (batch_size, channels)

        # Pass through MLP and combine
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = avg_out + max_out

        # Apply sigmoid to get channel weights
        scale = self.sigmoid(out).unsqueeze(2).unsqueeze(3)  # Shape: (batch_size, channels, 1, 1)

        # Multiply the input with the channel weights (broadcasting)
        return x * scale


class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.channel_attention = ChannelAttention

    def forward(self, rgb_features, hsv_features):
        # concatenate the feature maps together
        fused_features = torch.cat([rgb_features, hsv_features], dim=1)
        # conv layer to extrac relationships between the feature maps
        fused_features = self.conv(fused_features)
        # channel attention to select more relevant channels of H,S or V
        return self.channel_attention(fused_features)


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

    # Calculate the difference image
    diff_image = y0 - xt

    # Get model prediction
    predicted_noise = model(xt, t, y0=y0, diff_image=diff_image)

    # Calculate the loss
    loss = mse_loss(noise, predicted_noise)
    return loss


class Unet(nn.Module):
    """
    Proper UNet architecture with a deeper network that takes alot longer to run
    1 epoch = 24 minutes
    """

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
        """
        :param self:
        :param t: given random timesteps for the batch, usually of size batch_size
        :param channels: number of time dimensions which is also the columns in the vector
        :return: tensor of vector of time embeddings for each timestep where each row is the time embedding for 1 timestep
        """
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


"""
Create a simple UNet architecture to speed up computation time.
removed spatial attention
still has skip connections
still is basic UNet with up and down sample blocks and a bottle neck
still need to add channel attention
make and add content compensation module from the paper

consider making this UNet slightly deeper

"""


class SimpleUNet(nn.Module):

    def __init__(self, input_channels=3, out_channels=3, time_embedding=64, cond_inp_channels=3, device="cuda"):
        super().__init__()

        self.device = device
        self.time_embedding = time_embedding
        self.cond_out_channels = 3

        # Add ChannelAttention layers after downsampling and upsampling blocks
        self.ca_down1 = ChannelAttention(128)
        self.ca_down2 = ChannelAttention(256)
        self.ca_up1 = ChannelAttention(128)
        self.ca_up2 = ChannelAttention(64)

        # initial input layer if no conditional
        self.inc = DoubleConv(input_channels, 64)  # Initial convolution

        # conditional input convolutions
        self.cond_conv = nn.Conv2d(in_channels=cond_inp_channels, out_channels=self.cond_out_channels, kernel_size=1,
                                   bias=False)
        self.cat_cond_conv = nn.Conv2d(in_channels=input_channels + self.cond_out_channels, out_channels=64,
                                       kernel_size=3, padding=1)

        self.down1 = Down(64, 128, time_embedding)
        self.down2 = Down(128, 256, time_embedding)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 256)  # Simplified bottleneck

        # Decoder (up-sampling)
        self.up1 = Up(384, 128, time_embedding)
        self.up2 = Up(192, 64, time_embedding)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        :param self:
        :param t: given random timesteps for the batch, usually of size batch_size
        :param channels: number of time dimensions which is also the columns in the vector
        :return: tensor of vector of time embeddings for each timestep where each row is the time embedding for 1 timestep
        """
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y0=None):
        """

        :param x: pure gaussian noised image
        :param y0: ground truth image
        :param t: timestep
        :return: noise to remove
        """

        # if conditional image has been added
        if y0 is not None:
            # 1x1 conv to bring number of channels to 3
            cond_img = self.cond_conv(y0)
            # concatenate the noise x with the conditional input on channel dimension
            x = torch.cat([x, cond_img], dim=1)
            #
            x1 = self.cat_cond_conv(x)
        else:
            x1 = self.inc(x)

        # time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_embedding)

        x2 = self.down1(x1, t)
        x2 = self.ca_down1(x2)  # Apply channel attention after down sampling
        x3 = self.down2(x2, t)
        x3 = self.ca_down2(x3)  # Apply channel attention after down sampling
        # pdb.set_trace() # use pdb to diagnose problems in the code, check tensor shapes
        # Bottleneck
        x3 = self.bottleneck(x3)

        # Decoder
        x = self.up1(x3, x2, t)
        x = self.ca_up1(x)  # Apply channel attention after up-sampling

        x = self.up2(x, x1, t)
        x = self.ca_up2(x)  # Apply channel attention after up-sampling

        output = self.outc(x)
        return output

# tracking that the cpdm works correct, have to make sure i can get good results to get cpdm works
# get cpdm working first with the simpleunet
# compare with my RGB HSV one and hopefully get better results
# why is my HSV addition important, what is the difference of our work, what are the drawbacks of cpdm
# import pdb
# plot the loss after evey epoch for the conditional input
# check that the paired inputs are really paired, check everything
