# class to deal with retrieving data from the datasets for the model.
import os
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt
import torch
import torchvision
import cv2 as cv
import numpy as np


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path):
    grid = torchvision.utils.make_grid(images)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


# inherit pytorch dataset class
class DatasetUtils(Dataset):
    def __init__(self, root, transform, train=True, inference=False):
        self.root = root
        self.transform = transform
        self.train = train
        self.train = train

        # set up directories
        input_dir = os.path.join(root, "LSUD/LSUI/input")
        gt_dir = os.path.join(root, "LSUD/LSUI/GT")

        # sort the listed images
        self.input_images = sorted(os.listdir(input_dir))
        self.gt_images = sorted(os.listdir(gt_dir))

        # calculate training/test size
        total_size = len(self.input_images)
        train_size = int(0.85 * total_size)
        inference_size = int(0.05 * total_size)

        if inference:
            # get 5% for inference of gt and input dataset
            self.input_images = self.input_images[train_size:train_size + inference_size]
            self.gt_images = self.gt_images[train_size:train_size + inference_size]
        elif train:
            # get first 85% of the input and gt data
            self.input_images = self.input_images[:train_size]
            self.gt_images = self.gt_images[:train_size]
        elif not train:
            # get the last 10% of input and gt data for the Test set
            self.input_images = self.input_images[train_size + inference_size:]
            self.gt_images = self.gt_images[train_size + inference_size:]

    def __len__(self):
        return len(self.input_images)  # Or len(self.gt_images), since they should have the same length

    def __getitem__(self, item):

        # create the image paths for gt and input
        gt_image_path = os.path.join(self.root, "LSUD/LSUI/GT", self.gt_images[item])
        input_images_path = os.path.join(self.root, "LSUD/LSUI/input", self.gt_images[item])

        input_image = Image.open(input_images_path).convert('RGB')
        gt_image = Image.open(gt_image_path).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        image_np = gt_image.numpy().transpose(1, 2, 0)  # converts to numpys HWC format
        gt_hsv = cv.cvtColor(image_np, cv.COLOR_RGB2HSV)
        gt_hsv = torch.from_numpy(gt_hsv.transpose(2, 0, 1))

        return input_image, gt_image, gt_hsv

    def list_data(self):
        # Get the list of image filenames (assuming they have the same names in both directories)
        # print(self.input_images, "\n")
        # print(self.gt_images, "\n")

        # print the number of images in the directory
        print("There are ", len(self.input_images), " images in input directory")
        print("There are ", len(self.gt_images), " images in gt directory")

    def get_item(self):
        pass
