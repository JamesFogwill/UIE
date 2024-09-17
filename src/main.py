# main project loop

import os
import torch
import torchvision.utils
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import DatasetUtils
import NoiseSchedular
import Unet
import diff_tools as dt
import diff_modules as dm
import argparse

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

# Initialize SummaryWriter with the absolute path
writer = SummaryWriter(os.path.join(project_root, "runs", "logs0.4.5_testrgbagain"))

# Add a note/description for this specific run
writer.add_text("Notes", "Changes made in this run: made a few changes back to the inp layer and also increase output "
                         "if conditional to 6 channels and input of inp to 6 to try and retain that conditional "
                         "information"
                         "")

"""
All parameters
"""
# parameters

epochs = 26
img_size = 64  # 224 because random crop?
time_steps = 1000  # number of diffusion steps
batch_size = 8


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

READ THIS
You may run the training with either model shown below, they both work fine.
There is no saved state dict for the RGBHSV model, i ran out of time to make one.
The supplied state dict uses the SimpleUNet, you an run this in the file called Test.py
"""

# Create the Unet Model UNCOMMENT THE ONE YOU WANT TO USE. ONE AT A TIME.
model = Unet.SimpleUNet(input_channels=3, out_channels=3, time_embedding=64, pretrained_encoder=False, device="cuda").to(device)
# model = Unet.RGBHSVUNet(input_channels=3, out_channels=3, time_embedding=64, device="cuda").to(device)

noise_scheduler = NoiseSchedular.NoiseSchedular(img_size=img_size, time_steps=time_steps, beta_start=1e-4,
                                                beta_end=0.02,
                                                device="cuda").to(device)

# Create Dataset objects for train, inference and test
dataset_train = DatasetUtils.DatasetUtils(root="../", transform=training_transform, train=True, inference=False)
dataset_test = DatasetUtils.DatasetUtils(root="../", transform=test_transform, train=False, inference=False)
dataset_inference = DatasetUtils.DatasetUtils(root="../", transform=test_transform, train=False, inference=True)

# create the data loaders for the datasets
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
inference_loader = DataLoader(dataset_inference, batch_size=batch_size - 4, shuffle=False, drop_last=True)

# list the number of images in the directory for checking purposes
dataset_train.list_data()
dataset_test.list_data()

# verify batch is loading image pairs correctly
batch = next(iter(train_loader))
input_images, gt_images = batch

gt_grid = torchvision.utils.make_grid(gt_images.clip(0, 1), nrow=batch_size)
input_grid = torchvision.utils.make_grid(input_images.clip(0, 1), nrow=batch_size)

writer.add_image('Input images', input_grid)
writer.add_image('Ground truth images', gt_grid)

"""
Main Training loop start
"""
# create loss and optimiser
optimiser = optim.AdamW(model.parameters(), lr=0.0005)
mse = nn.MSELoss()

# create sample time-steps and sample model to add to tensor board
t = noise_scheduler.sample_timesteps(batch_size).to(device)
writer.add_graph(model, (input_images.to(device), t,))

writer.close()

total_batches = len(train_loader)
total_loss = 0.0
first = True

for epoch in range(epochs):

    print("starting epoch, ", epoch + 1)

    for batch_idx, batch in enumerate(train_loader):
        # unpacks the batch
        input_images, gt_images = batch

        # send all images to the currently selected device / GPU
        gt_images = gt_images.to(device)
        input_images = input_images.to(device)

        # Comment When Running - display first image in the input and gt batch to check the image pairs are set up
        # dt.display_image_pair(gt_images, gt_images)

        # create timestep tensor
        t = noise_scheduler.sample_timesteps(batch_size).to(device)

        # return all the noised images and the noise added to each image
        x_t, noise = noise_scheduler.noise_images(gt_images, t)

        # run reverse diffusion for to get predicted noise with ground truth images to guide.
        predicted_noise = model(x_t, t, y0=input_images)

        # Work out loss
        loss = mse(noise, predicted_noise)
        total_loss += loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # works out the average loss per epoch and prints to the terminal + tensorboard then resets the loss
    print(f'Average Epoch{epoch + 1} Loss:[{total_loss / total_batches}]')
    writer.add_scalar("Average Epoch Training Loss", total_loss / total_batches, epoch)
    total_loss = 0.0

    # sample new images the first and every 25 epochs where n is the number of sampled images
    if (epoch + 1) % 25 == 0 or first is True:
        # save the model every 25 epochs
        torch.save(model.state_dict(),
                   os.path.join("../", "results", "models", "checkpoints", f"ckpt_epoch_{epoch + 1}.pt"))

        print(f'Running Inference Epoch{epoch + 1}')
        first = False
        dt.inference_full(inference_loader=inference_loader, model=model, noise_scheduler=noise_scheduler, time_steps=time_steps, writer=writer, epoch=epoch)

        # Sample images for inference and add to tensorboard
        # sampled_images = noise_scheduler.generate_images(model, n=1, inf_loader=inference_loader)
