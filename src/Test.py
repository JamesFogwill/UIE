import os

import torch
import torchvision.utils
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import DatasetUtils
import NoiseSchedular
import Unet
import diff_tools as dt

"""

    Hello, if you would like to run the testing then you will have to download the LSUI dataset as there are too many images for github to store.
    The model state dictionary is stored in the ResultsGithub folder and it works within this file.
    you will have to configure the project root,
    start a tenosr board,
    There is no testing model for the RGBHSV diffusion model Yet! Coming soon

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
writer = SummaryWriter(os.path.join(project_root, "runs", "logs1.0_Get_Results"))

# init noise schedular
noise_scheduler = NoiseSchedular.NoiseSchedular(img_size=img_size, time_steps=time_steps, beta_start=1e-4,
                                                beta_end=0.02,
                                                device="cuda").to(device)

# Create the Unet Model for reverse diffusion
model = Unet.SimpleUNet(input_channels=3, out_channels=3, time_embedding=64, pretrained_encoder=False, device="cuda").to(device)
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
checkpoint_path = os.path.join("../", "ResultsGithub", "Model state dict", "WorkingSimpleUnet", "ckpt_epoch_300.pt")
model.load_state_dict(torch.load(checkpoint_path))

# Set the model to evaluation mode for testing
model.eval()

# Generate images using the test loader
with torch.no_grad():

    # generate images using test loader and evaluation metrics for them
    # sampled_images_metrics = noise_scheduler.generate_images_metrics(model, batch_size=8, loader=test_loader, writer=writer)
    # sampled_images, avg_psnr, avg_ssim, avg_mse = sampled_images_metrics

    dt.inference_full(inference_loader=test_loader, model=model, noise_scheduler=noise_scheduler, time_steps=time_steps, writer=writer, epoch=300, testing=True)

    # Add metrics to TensorBoard
    #writer.add_scalar('PSNR', avg_psnr)
    #writer.add_scalar('SSIM', avg_ssim)
    #writer.add_scalar('MSE', avg_mse)

# Save or visualize the generated images
#dt.save_images(images=sampled_images, path=os.path.join("../", "results", "test_images", "generated.jpg"))
