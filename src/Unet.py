import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import diff_tools as dt
import diff_modules as dm
import pdb
import random
import kornia
import torchvision.models as models


class DoubleConv(nn.Module):
    # TAKEN FROM https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
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


class Down(nn.Module):
    # TAKEN FROM https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
    def __init__(self, in_channels, out_channels, time_embedding_dim=64):
        """
            Initializes the Down block.

            Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.
                time_embedding_dim (int): Dimensionality of the time embedding vector.
        """
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),  # Downsamples the feature map by a factor of 2 in both spatial dimensions
            DoubleConv(in_channels, in_channels, residual=True),
            # Applies two convolutional layers with a residual connection
            DoubleConv(in_channels, out_channels)
            # Applies two more convolutional layers to change the number of channels
        )

        self.time_embedding_layer = nn.Sequential(
            nn.SiLU(),  # sigmoid linear unit activation function
            nn.Linear(
                # Linear layer to project the time embedding to the correct size, enabling it to be added to the
                # feature map for time conditioning
                time_embedding_dim,
                out_channels),
        )

    def forward(self, x, t):
        """
            Forward pass of the Down block.

            Args:
                x (torch.Tensor): Input feature map.
                t (torch.Tensor): Time embedding vector.

            Returns:
                torch.Tensor: Downsampled and time-conditioned feature map.
        """

        x = self.down_sample(x)  # downsamples the input feature map

        embedding = self.time_embedding_layer(t)  # Project time embedding to the output channel space
        embedding = embedding[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[
            -1])  # Expand the time embedding to match the spatial dimensions of 'x'

        # Add the time embedding to the downsampled feature map
        return x + embedding


class Up(nn.Module):
    # TAKEN FROM https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
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


class ChannelAttention(nn.Module):
    # reduction ration controlling the dimensionality reduction in the bottleneck MLP
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        # reduces the spatial dimensions of the input to 1x1, effectively computing the global average of each channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        # Creates an Adaptive Max Pooling layer, computes the global maximum for each channel
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling

        # Bottleneck MLP to learn channel-wise dependencies
        self.mlp = nn.Sequential(
            # reduces the dimensionality by reduction_ratio
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.GELU(),
            # restores the original dimensionality (in_channels)
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

        # generate attention weights between 0 and 1
        self.sigmoid = nn.Sigmoid()  # To produce attention weights between 0 and 1

    def forward(self, x):
        # Get global average and max pooled features
        avg_pool = self.avg_pool(x).squeeze(-1).squeeze(-1)  # Shape: (batch_size, channels)
        max_pool = self.max_pool(x).squeeze(-1).squeeze(-1)  # Shape: (batch_size, channels)

        # Pass both features through MLP and combine element wise
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = avg_out + max_out

        # Apply sigmoid to get channel weights
        scale = self.sigmoid(out).unsqueeze(2).unsqueeze(3)  # Shape: (batch_size, channels, 1, 1)

        # Multiply the input with the channel weights (broadcasting)
        return x * scale


class RGBHSVUNet(nn.Module):

    def __init__(self, input_channels=3, out_channels=3, time_embedding=64, cond_inp_channels=6, device="cuda"):
        super().__init__()

        self.device = device
        self.time_embedding = time_embedding
        self.cond_out_channels = 3

        # Add ChannelAttention layers after down-sampling and up-sampling blocks
        self.ca_down1 = ChannelAttention(128)
        self.ca_down2 = ChannelAttention(256)
        self.ca_down3 = ChannelAttention(256)

        self.ca_up1 = ChannelAttention(128)
        self.ca_up2 = ChannelAttention(64)
        self.ca_up3 = ChannelAttention(64)

        # conditional input convolutions
        self.cond_conv = nn.Conv2d(in_channels=cond_inp_channels, out_channels=self.cond_out_channels, kernel_size=1,
                                   bias=False)
        self.cat_cond_conv = nn.Conv2d(in_channels=input_channels + self.cond_out_channels, out_channels=64,
                                       kernel_size=3, padding=1)

        self.cond_conv_rgb = nn.Conv2d(in_channels=cond_inp_channels, out_channels=self.cond_out_channels,
                                       kernel_size=1,
                                       bias=False)
        self.cond_conv_hsv = nn.Conv2d(in_channels=cond_inp_channels, out_channels=self.cond_out_channels,
                                       kernel_size=1,
                                       bias=False)
        self.cat_cond_conv_rgb = nn.Conv2d(in_channels=input_channels + self.cond_out_channels, out_channels=6,
                                           kernel_size=3, padding=1)
        self.cat_cond_conv_hsv = nn.Conv2d(in_channels=input_channels + self.cond_out_channels, out_channels=6,
                                           kernel_size=3, padding=1)

        self.no_cond = nn.Conv2d(in_channels=input_channels, out_channels=input_channels + 3, kernel_size=3, padding=1)

        self.no_cond_hsv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels + 3, kernel_size=3, padding=1)

        # self.fusion = FeatureFusion(512, 256)
        self.SKC = SKC(256, 256)

        # encoder (down-sampling)
        self.inc = DoubleConv(input_channels+3, 64)  # Initial convolution
        self.down1 = Down(64, 128, time_embedding)
        self.down2 = Down(128, 256, time_embedding)
        self.down3 = Down(256, 256, time_embedding)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 512)  # Simplified bottleneck
        self.bottleneck2 = DoubleConv(512, 256)  # Simplified bottleneck

        # Decoder (up-sampling)
        self.up1 = Up(512, 128, time_embedding)
        self.up2 = Up(256, 64, time_embedding)
        self.up3 = Up(128, 64, time_embedding)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        # ADAPTED FROM https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
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
        :param y0: the rgb input image, gt when training
        :param x: pure gaussian noised image
        :param t: timestep
        :return: noise to remove
        """

        # if conditional image has been added
        if y0 is not None:

            y0_hsv = kornia.color.rgb_to_hsv(y0)
            # difference image for both
            diff_img_rgb = y0 - x
            diff_img_hsv = y0_hsv - x

            # concatenate difference image for both rgb and hsv
            conditional_rgb = torch.cat([y0, diff_img_rgb], dim=1)
            conditional_hsv = torch.cat([y0_hsv, diff_img_hsv], dim=1)

            # 1x1 conv to bring number of channels to 3 for both colour spaces
            cond_img_rgb = self.cond_conv_rgb(conditional_rgb)
            cond_img_hsv = self.cond_conv_hsv(conditional_hsv)

            # concatenate noise with conditional inputs
            x_rgb = torch.cat([x, cond_img_rgb], dim=1)
            x_hsv = torch.cat([x, cond_img_hsv], dim=1)

            # final concat convolution
            x = self.cat_cond_conv_rgb(x_rgb)
            x_hsv = self.cat_cond_conv_hsv(x_hsv)

        else:
            y0_hsv = kornia.color.rgb_to_hsv(x)
            x = self.no_cond(x)
            x_hsv = self.no_cond_hsv(y0_hsv)

        # time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_embedding)

        x1 = self.inc(x)
        # encoder rgb
        x2_rgb = self.down1(x1, t)
        x2_rgb = self.ca_down1(x2_rgb)  # Apply channel attention after down sampling
        x3_rgb = self.down2(x2_rgb, t)
        x3_rgb = self.ca_down2(x3_rgb)  # Apply channel attention after down sampling
        x4_rgb = self.down3(x3_rgb, t)
        x4_rgb = self.ca_down3(x4_rgb)

        x1_hsv = self.inc(x_hsv)
        # encoder hsv
        x2_hsv = self.down1(x1_hsv, t)
        x2_hsv = self.ca_down1(x2_hsv)  # Apply channel attention after down sampling
        x3_hsv = self.down2(x2_hsv, t)
        x3_hsv = self.ca_down2(x3_hsv)  # Apply channel attention after down sampling
        x4_hsv = self.down3(x3_hsv, t)
        x4_hsv = self.ca_down3(x4_hsv)

        # Bottleneck
        # x_fused = self.fusion(x4_rgb, x4_hsv)
        x_SKC = self.SKC(x4_rgb, x4_hsv)
        x4 = self.bottleneck(x_SKC)
        x4 = self.bottleneck2(x4)

        # Decoder
        x = self.up1(x4, x3_rgb, t)
        x = self.ca_up1(x)  # Apply channel attention after up-sampling

        x = self.up2(x, x2_rgb, t)
        x = self.ca_up2(x)  # Apply channel attention after up-sampling

        x = self.up3(x, x1, t)
        x = self.ca_up3(x)

        output = self.outc(x)
        return output


# compare with my RGB HSV one and hopefully get better results
# why is my HSV addition important, what is the difference of our work, what are the drawbacks of cpdm
# import pdb
# plot the loss after evey epoch for the conditional input
# check that the paired inputs are really paired, check everything

class SimpleUNet(nn.Module):

    def __init__(self, input_channels=3, out_channels=3, time_embedding=64, cond_inp_channels=6,
                 pretrained_encoder=True, device="cuda"):
        super().__init__()

        self.device = device
        self.time_embedding = time_embedding
        self.cond_out_channels = 3
        self.pretrained_encoder = pretrained_encoder

        # Add ChannelAttention layers after downsampling and upsampling blocks
        self.ca_down1 = ChannelAttention(128)
        self.ca_down2 = ChannelAttention(256)
        self.ca_down3 = ChannelAttention(256)

        self.ca_up1 = ChannelAttention(128)
        self.ca_up2 = ChannelAttention(64)
        self.ca_up3 = ChannelAttention(64)

        # image encoder
        if pretrained_encoder:
            resnet = models.resnet18(pretrained=True)
            modules = list(resnet.children())[:-1]  # Remove the last layer
            self.image_encoder = nn.Sequential(*modules)
            self.image_encoder.eval()  # Set to evaluation mode
            self.projection = nn.Linear(512, time_embedding)  # Adjust 512 if using a different ResNet
        else:
            # self.image_encoder = dm.ImageEncoder(c_in=3, embedding_dim=64)
            pass

        # conditional input convolutions
        self.cond_conv = nn.Conv2d(in_channels=cond_inp_channels, out_channels=self.cond_out_channels, kernel_size=1,
                                   bias=False)
        self.cat_cond_conv = nn.Conv2d(in_channels=input_channels + self.cond_out_channels, out_channels=input_channels + self.cond_out_channels,
                                       kernel_size=3, padding=1)
        self.no_cond = nn.Conv2d(in_channels=input_channels, out_channels=input_channels+3, kernel_size=3, padding=1)

        # encoder (down-sampling)
        self.inp = dm.DoubConvBlock(input_channels+3, 64)

        self.down1 = Down(64, 128, time_embedding)
        self.down2 = Down(128, 256, time_embedding)
        self.down3 = Down(256, 256, time_embedding)

        # Bottleneck
        self.bottleneck = dm.DoubConvBlock(256, 512)  # Simplified bottleneck
        self.bottleneck2 = dm.DoubConvBlock(512, 256)  # Simplified bottleneck

        # Decoder (up-sampling)
        self.up1 = Up(512, 128, time_embedding)
        self.up2 = Up(256, 64, time_embedding)
        self.up3 = Up(128, 64, time_embedding)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        :param self:
        :param t: given random timesteps for the batch, usually of size batch_size
        :param channels: number of time dimensions which is also the columns in the vector
        :return: tensor of vector of time embeddings for each timestep where each row is the time embedding for 1 timestep

        This module was created by
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
        :param x: pure gaussian noised image [batch,3, 64,64]
        :param y0: conditional input [batch, 3, 64,64]
        :param t: timestep
        :return: noise to remove
        """
        testing = False
        # time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_embedding)

        # experimental concatenation to the time embedding after using pretrained model for image encoder
        if y0 is not None and testing is True:
            if self.pretrained_encoder:  # Only apply no_grad if using pretrained encoder
                with torch.no_grad():
                    image_embedding = self.image_encoder(y0)
            else:
                image_embedding = self.image_encoder(y0)  # Allow gradients to flow for custom encoder

            # Flatten the spatial dimensions and project to match time_embedding_dim
            image_embedding = image_embedding.squeeze()
            image_embedding = self.projection(image_embedding)  # Shape: (batch_size, time_embedding_dim)

            # Combine time and conditional embeddings (concatenation)
            t = torch.cat([t, image_embedding], dim=1)
            # concatenate the difference image and the ground truth image output: [batch,6,64,64]
            x = torch.cat([x, y0], dim=1)  # [batch,6,64,64]

        # 2. Conditional Embedding commented out for test
        #if y0 is not None and testing is True:
            #image_embedding = self.image_encoder(y0)  # Encode the image
            #t += image_embedding  # Add image embedding to timestep embedding

        # if conditional image has been added
        if y0 is not None:
            # get the difference image
            diff_image = y0 - x  # [batch,3,64,64]
            # concatenate the difference image and the ground truth image output: [batch,6,64,64]
            conditional = torch.cat([y0, diff_image], dim=1)  # [batch,6,64,64]
            # 1x1 conv to bring number of channels to output [batch,3,64,64]
            cond_img = self.cond_conv(conditional)  # [batch,3,64,64]
            # concatenate the noise x with the conditional input on channel dimension
            x = torch.cat([x, cond_img], dim=1)  # [batch,6,64,64]

            x = self.cat_cond_conv(x)  # [batch,6,64,64]
        else:
            # even tho this isn't trained it will provide inc with the right input?
            x = self.no_cond(x)

        x1 = self.inp(x)
        # encoder
        x2 = self.down1(x1, t)
        x2 = self.ca_down1(x2)  # Apply channel attention after down sampling
        x3 = self.down2(x2, t)
        x3 = self.ca_down2(x3)  # Apply channel attention after down sampling
        x4 = self.down3(x3, t)
        x4 = self.ca_down3(x4)

        # pdb.set_trace() # use pdb to diagnose problems in the code, check tensor shapes
        # Bottleneck
        x4 = self.bottleneck(x4)
        x4 = self.bottleneck2(x4)

        # Decoder
        x = self.up1(x4, x3, t)
        x = self.ca_up1(x)
        x = self.up2(x, x2, t)
        x = self.ca_up2(x)
        x = self.up3(x, x1, t)
        x = self.ca_up3(x)

        output = self.outc(x)
        return output
