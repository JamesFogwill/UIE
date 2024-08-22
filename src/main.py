# main project loop

import os
import sys
import torch
import torchvision.utils
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import DatasetUtils
import NoiseSchedular
import diff_modules as dm
import Unet

import matplotlib.pyplot as plt

"""
Set up device
"""
# device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda is available")
else:
    device = torch.device("cpu")
    print("Cuda not available, training on cpu")

# Absolute path to the project root (UIE folder)
project_root = r"C:\Users\james\OneDrive\Documents\GitHub\UIE"

"""
All parameters
"""
# parameters

epochs = 501
img_size = 64  # 224 because random crop?
time_steps = 1000  # number of diffusion steps
batch_size = 8

# Initialize SummaryWriter with the absolute path
writer = SummaryWriter(os.path.join(project_root, "runs", "logs0.12_cond"))
"""
Training transforms.

RHF flips the image horizontally with float(chance 0-1)
RC crop the image to 224x224
RR rotates the image by 15 degrees
CJ applies colour augmentation to the image

In a diffusion model its common practice to normalise using lambda to scale image pixel values to the -1,1 range
This aligns with the noise distribution (gaussian)

"""
training_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                         transforms.RandomHorizontalFlip(0.5),
                                         transforms.RandomRotation(15),
                                         transforms.ToTensor(),  # Scales data into [0,1]
                                         transforms.Lambda(
                                             lambda t: (t * 2) - 1)])  # Shifts the scale to [-1,1], normalises

# transforms for test data
test_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""
Creation of model, dataset and noise schedular Objects
"""
noise_scheduler = NoiseSchedular.NoiseSchedular(img_size=img_size, time_steps=time_steps, beta_start=1e-4, beta_end=0.02,
                                                device="cuda").to(device)

# Create the Unet Model for reverse diffusion
model = Unet.SimpleUNet(input_channels=3, out_channels=3, time_embedding=64, device="cuda").to(device)

# Create Dataset objects for train, inference and test
dataset_train = DatasetUtils.DatasetUtils(root="../", transform=training_transform, train=True, inference=False)
dataset_test = DatasetUtils.DatasetUtils(root="../", transform=test_transform, train=False, inference=False)
dataset_inference = DatasetUtils.DatasetUtils(root="../", transform=test_transform, train=False, inference=True)

# create the data loaders for the datasets
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
inference_loader = DataLoader(dataset_inference, batch_size=batch_size, shuffle=False, drop_last=True)

# list the number of images in the directory for checking purposes
dataset_train.list_data()
dataset_test.list_data()

"""
Show the batch image pairs to verify correct loading.

Uses torchvision grids to group input and gt images in a horizontal line
Uses matplotlib subplots to plot the two grids together on a grid
Uses tensorboard to save the first batch to tensorboard

"""

# create and unpack batch
batch = next(iter(train_loader))
input_images, gt_images, gt_hsv = batch

# set up the figure using torchvision
gt_grid = torchvision.utils.make_grid(gt_images.clip(0, 1), nrow=batch_size)
input_grid = torchvision.utils.make_grid(input_images.clip(0, 1), nrow=batch_size)

"""
# permute them so values are as expected for plt.subplots
gt_grid_np = gt_grid.permute(1, 2, 0).numpy()
input_grid_np = input_grid.permute(1, 2, 0).numpy()

# matplot sub-plot to arrange the torch grids
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

# set up the axis
axes[0].imshow(input_grid_np)
axes[0].set_title("Input Images")
axes[0].axis('off')

axes[1].imshow(gt_grid_np)
axes[1].set_title("GT images")
axes[1].axis('off')

# arranges the grid and shows the grid
plt.tight_layout()
plt.show()
"""

# add input and gt images to tensor board
writer.add_image('Input images', input_grid)
writer.add_image('Ground truth images', gt_grid)

"""
The forward diffusion loop.

contains the forward diffusion/noising in a for loop.
Model learns how to recreate the up-scaled images without the water.

"""
# create loss and optimiser
optimiser = optim.AdamW(model.parameters(), lr=0.001)
mse = nn.MSELoss()

# create sample time-steps and sample model to add to tensor board
t = noise_scheduler.sample_timesteps(batch_size).to(device)
writer.add_graph(model, (input_images.to(device), t))

writer.close()

total_batches = len(train_loader)
total_loss = 0.0

for epoch in range(epochs):
    total_loss = 0
    print("starting epoch, ", epoch+1)

    for batch_idx, batch in enumerate(train_loader):

        # unpacks the batch
        input_images, gt_images, gt_hsv = batch

        # send all images to the currently selected device / GPU
        gt_images = gt_images.to(device)
        gt_hsv = gt_hsv.to(device)
        input_images = input_images.to(device)

        # Comment When Running - display first image in the input and gt batch to check the image pairs are set up
        # dm.display_image_pair(input_images, gt_images)

        # create timestep tensor
        t = noise_scheduler.sample_timesteps(batch_size).to(device)

        # return all the noisy images and the noise added to each image
        x_t, noise = noise_scheduler.noise_images(input_images, t)

        # run reverse diffusion for to get predicted noise with ground truth images to guide.
        predicted_noise = model(x_t, t, y0=gt_images)

        # Work out loss
        loss = mse(noise, predicted_noise)
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch[{epoch + 1}/{epochs}], Batch[{batch_idx}/{total_batches}], Loss[{loss.item():.4f}]')
            writer.add_scalar("Training loss", total_loss / 10, epoch * total_batches + batch_idx)
            total_loss = 0.0
        # use tensor board helps visualise loss, network.

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # sample new images every 10 epochs where n is the number of sampled images
    if (epoch + 1) % 25 == 0:

        # Sample images for inference and add to tensorboard
        # sampled_images = noise_scheduler.generate_images(model, n=gt_images.shape[0], loader=inference_loader)
        sampled_images = noise_scheduler.generate_images2(model, n=1, inf_loader=inference_loader)
        sampled_images_grid = torchvision.utils.make_grid(sampled_images)
        writer.add_image(f'Epoch[{epoch + 1}] Sampled images', sampled_images_grid)

        # save images to folders in the root directory of project
        DatasetUtils.save_images(images=sampled_images, path=os.path.join("../", "results", "sample_images", f"{epoch}.jpg"))
        # save the model state_dict for future use
        torch.save(model.state_dict(), os.path.join("../", "results", "models", "checkpoints", f"ckpt_epoch_{epoch + 1}.pt"))

"""
Testing of the model
"""
# Load the state dictionary from the specified checkpoint
checkpoint_path = os.path.join("../", "results", "models", "checkpoints", "ckpt_epoch_500.pt")
model.load_state_dict(torch.load(checkpoint_path))

# Set the model to evaluation mode for testing
model.eval()

# Generate images using the test loader
with torch.no_grad():

    # generate 3 images for each gt image in the test set
    sampled_images = noise_scheduler.generate_images2(model, n=3, inf_loader=test_loader)

    # write these images to the tensor board
    sampled_images_grid = torchvision.utils.make_grid(sampled_images)
    writer.add_image(f'Test Sample Images', sampled_images_grid)

# Save or visualize the generated images
DatasetUtils.save_images(images=sampled_images, path=os.path.join("../", "results", "test_images", "generated.jpg"))
