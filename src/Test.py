import os

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

"""
Set up

Sets up the device:
    Cuda

Sets up variables:
    img_size
    time_steps
    batch_size

Sets up objects:
    Summary Writer
    Model
    NoiseSchedular
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

img_size = 64  # 224 because random crop?
time_steps = 1000  # number of diffusion steps
batch_size = 8

# Initialize SummaryWriter with the absolute path
writer = SummaryWriter(os.path.join(project_root, "runs", "logs0.13_cond"))

# init noise schedular
noise_scheduler = NoiseSchedular.NoiseSchedular(img_size=img_size, time_steps=time_steps, beta_start=1e-4, beta_end=0.02,
                                                device="cuda").to(device)

# Create the Unet Model for reverse diffusion
model = Unet.SimpleUNet(input_channels=3, out_channels=3, time_embedding=64, device="cuda").to(device)

"""
Creates test transform and dataset loaders
"""

# transforms for test data
test_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Sets Up Test Loader
dataset_test = DatasetUtils.DatasetUtils(root="../", transform=test_transform, train=False, inference=False)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

"""
Testing of the model
"""
# Load the state dictionary from the specified checkpoint
checkpoint_path = os.path.join("../", "results", "models", "checkpoints", "ckpt_epoch_250.pt")
model.load_state_dict(torch.load(checkpoint_path))

# Set the model to evaluation mode for testing
model.eval()

# Generate images using the test loader
with torch.no_grad():

    # generate 3 images for each gt image in the test set
    sampled_images = noise_scheduler.generate_images2(model, n=1, inf_loader=test_loader)

    # write these images to the tensor board
    sampled_images_grid = torchvision.utils.make_grid(sampled_images)
    writer.add_image(f'Test Sample Images', sampled_images_grid)

# Save or visualize the generated images
DatasetUtils.save_images(images=sampled_images, path=os.path.join("../", "results", "test_images", "generated.jpg"))
