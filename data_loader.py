import os
import numpy as np
import torch
import cv2
from PIL import Image
import albumentations as A
from skimage import color
import random
import torch.nn as nn


np.random.seed(37)  # Set seed for reproducibility


def stain_augmentation(img, theta=0.03):
    th = 0.9  # Threshold for background detection
    alpha = np.random.uniform(1 - theta, 1 + theta, (1, 3))  # Random scaling factors
    beta = np.random.uniform(-theta, theta, (1, 3))  # Random perturbations
    img = np.array(img)

    gray_img = color.rgb2gray(img)  # Convert to grayscale
    background = (gray_img > th)  # Identify background
    background = background[:, :, np.newaxis]  # Expand dimensions for broadcasting

    s = color.rgb2hed(img)  # Convert RGB to HED color space
    ns = alpha * s + beta  # Apply perturbations in HED space
    nimg = color.hed2rgb(ns)  # Convert back to RGB

    # Rescale the image
    imin = nimg.min()
    imax = nimg.max()
    rsimg = ((nimg - imin) / (imax - imin))  # Normalize to [0, 1]
    rsimg = (1 - background) * rsimg + background * img / 255  # Blend with original image

    rsimg = (255 * rsimg).astype('uint8')  # Convert back to uint8
    return rsimg


# Augmentation pipeline for training
aug = A.Compose([
    A.RandomCrop(1024, 1024),  # Randomly crop to 1024x1024
    A.VerticalFlip(p=0.5),  # Random vertical flip
    A.HorizontalFlip(p=0.5),  # Random horizontal flip
    A.RandomRotate90(p=0.5),  # Random 90-degree rotation
    A.ShiftScaleRotate(shift_limit=(-0.1, 0.1), scale_limit=(-0.3, 0.3), rotate_limit=(-15, 15), p=0.5),
    A.ColorJitter(),  # Randomly change brightness, contrast, saturation
    A.RandomBrightnessContrast(p=0.2),  # Random brightness and contrast
    A.HueSaturationValue(p=0.6),  # Randomly change hue and saturation
    A.RandomGamma(p=0.6)  # Randomly change gamma
])

# Augmentation pipeline for validation
Val_aug = A.Compose([
    A.RandomCrop(512, 512),  # Randomly crop to 512x512 for validation
])


def data_mean_normalization(im):
    im_ = im.astype("float32")  # Convert to float32
    # Subtract channel-wise means
    im_ -= np.array((0.485, 0.456, 0.406))
    # Divide by channel-wise standard deviations
    im_ /= np.array((0.229, 0.224, 0.225))
    return im_


class DataGenerator(torch.utils.data.Dataset):
    """Generates data for PyTorch"""

    def __init__(self, root, split, batch_size=32, dim=(32, 32), n_channels=3,
                 n_classes=10, shuffle=True):
        """Initialization"""
        self.root = root
        self.split = split
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        # Define directories for images and labels
        self.image_dir = os.path.join(self.root, self.split, self.split + '_data_all')
        self.label_dir = os.path.join(self.root, self.split, self.split + '_labels_all')

        # Load file list for segmentation
        file_list = os.path.join(self.root, self.split, self.split + '_segmentation', self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
        self.indexes = np.arange(len(self.files))
        # self.on_epoch_end()  # Uncomment if needed for shuffling

    def __len__(self):
        return len(self.files)  # Number of samples

    def __getitem__(self, index: int):
        """Generate one sample of data"""
        indexes = self.indexes[index]
        list_IDs_temp = self.files[indexes]

        # Load image and label
        image_path = os.path.join(self.image_dir, list_IDs_temp + '.jpg')
        label_path = os.path.join(self.label_dir, list_IDs_temp + '_modlabel.jpg')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)  # Read label in grayscale

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        # Apply augmentations
        augmented = aug(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = image / 255  # Normalize image to [0, 1]
        image = data_mean_normalization(image)  # Apply mean normalization
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
        image = image.astype(np.float32)  # Ensure float32 type
        label = label.astype(np.longlong)  # Convert label to longlong type

        image = np.rollaxis(image, -1, 0)  # Change shape to (C, H, W)
        # Store class labels
        label = binarylabel(label, self.n_classes)  # Convert to binary labels
        label = np.rollaxis(label, -1, 0)  # Change shape to (C, H, W)

        # Typecasting
        x, y = torch.tensor(image), torch.tensor(label)
        return x, y


def binarylabel(im_label, classes):
    """Convert label to binary format"""
    im_dims = im_label.shape
    lab = np.zeros([im_dims[0], im_dims[1], classes], dtype="uint8")  # Initialize label array
    for class_index in range(classes):
        lab[im_label == class_index, class_index] = 1  # Set binary class labels
    return lab


class ValDataGenerator(torch.utils.data.Dataset):
    """Generates validation data for PyTorch"""

    def __init__(self, root, split, batch_size=32, dim=(32, 32), n_channels=3,
                 n_classes=10, shuffle=True):
        """Initialization"""
        self.root = root
        self.split = split
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        # Define directories for images and labels
        self.image_dir = os.path.join(self.root, self.split, self.split + '_data_all')
        self.label_dir = os.path.join(self.root, self.split, self.split + '_labels_all')

        # Load file list for segmentation
        file_list = os.path.join(self.root, self.split, self.split + '_segmentation', self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]
        self.indexes = np.arange(len(self.files))
        # self.on_epoch_end()  # Uncomment if needed for shuffling

    def __len__(self):
        return len(self.files)  # Number of validation samples

    def __getitem__(self, index: int):
        """Generate one validation sample"""
        indexes = self.indexes[index]
        list_IDs_temp = self.files[indexes]

        # Load image and label
        image_path = os.path.join(self.image_dir, list_IDs_temp + '.jpg')
        label_path = os.path.join(self.label_dir, list_IDs_temp + '_modlabel.jpg')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)  # Read label in grayscale

        # Apply validation augmentations
        augmented = Val_aug(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = image / 255  # Normalize image to [0, 1]
        image = data_mean_normalization(image)  # Apply mean normalization
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
        image = image.astype(np.float32)  # Ensure float32 type
        label = label.astype(np.longlong)  # Convert label to longlong type

        image = np.rollaxis(image, -1, 0)  # Change shape to (C, H, W)

        # Store class labels
        label = binarylabel(label, self.n_classes)  # Convert to binary labels
        label = np.rollaxis(label, -1, 0)  # Change shape to (C, H, W)

        # Typecasting
        x, y = torch.tensor(image), torch.tensor(label)
        return x, y
