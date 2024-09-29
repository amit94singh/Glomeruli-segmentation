import sys
import numpy as np
import os
from data_loader import DataGenerator, ValDataGenerator
import segmentation_models_pytorch as smp
import torch
from trainer import Trainer
from transformers import SegformerForSemanticSegmentation
from torch.utils.data.sampler import SubsetRandomSampler
from Losses import ERS
from torchsummary import summary
from networks import SegNext

# Set random seeds for reproducibility
np.random.seed(122)
torch.manual_seed(122)

# Enable mixed precision training
from torch.cuda.amp import GradScaler, autocast

# Configuration
gpu_id = 0  # ID of the GPU to use
savepath = '/mnt/prj002/glomerulus/training_code/'  # Path to save model and logs
root_folder = '/mnt/prj002/glomerulus/Splited/'  # Path to the dataset
classes = 2  # Number of classes in the dataset

# Define image dimensions and batch size
batchsize = 16
depth = 3  # Number of input channels (RGB)
height = 1024  # Height of the input images
width = 1024  # Width of the input images

# Pre-trained model settings (commented out examples)
preTrained = None
pretrained_model_name = "nvidia/mit-b1"
id2label = {0: 'Background', 1: 'glomerulus'}  # Mapping of class indices to labels
label2id = {v: k for k, v in id2label.items()}  # Inverse mapping

# Initialize the model (uncomment to use a different model)


# model = smp.Unet(
#         encoder_name='timm-efficientnet-b4',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7 or timm-efficientnet-b4
#         encoder_weights='noisy-student',  # use imagenet or 'noisy-student' pre-trained weights for encoder initialization
#         in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#         classes=classes,  # model output channels (number of classes in your dataset)
#     )

# model = SegformerForSemanticSegmentation.from_pretrained(
#     pretrained_model_name,
#     num_labels=classes,
#     id2label=id2label,
#     label2id=label2id
# )

# Initialize the SegNext model
model = SegNext(num_classes=classes)

# Initialize the GradScaler for mixed precision training
scaler = GradScaler()

# Parameters for data loading
params = {
    'dim': (height, width),  # Image dimensions
    'batch_size': batchsize,  # Batch size for training
    'n_classes': classes,  # Number of classes
    'n_channels': depth,  # Number of input channels
    'shuffle': True  # Shuffle the dataset
}

# Load model weights if pre-trained weights are specified
if preTrained:
    model_weights = torch.load(preTrained, map_location='cuda:' + str(gpu_id))
    model.load_state_dict(model_weights)

# Data generators for training and validation
training_generator = DataGenerator(root_folder, 'train', **params)  # Custom data generator for training
validation_generator = ValDataGenerator(root_folder, 'val', **params)  # Custom data generator for validation

# Wrap data generators in DataLoader for batching
training_generator = torch.utils.data.DataLoader(training_generator, batch_size=batchsize, num_workers=16, drop_last=True)
validation_generator = torch.utils.data.DataLoader(validation_generator, batch_size=20, num_workers=16, drop_last=True)

print("Compiling Model...")

# Initialize the loss function and optimizer
criterion = ERS()  # Instantiate the combined loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)  # AdamW optimizer with a small learning rate

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.00001,
    epochs=120,
    steps_per_epoch=1,
    pct_start=0.02,
    verbose=False
)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    device=gpu_id,
    criterion=criterion,
    optimizer=optimizer,
    training_DataLoader=training_generator,
    validation_DataLoader=validation_generator,
    lr_scheduler=scheduler,
    epochs=100,  # Total number of epochs to train
    epoch=0,  # Starting epoch
    notebook=True,  # If running in a notebook, set to True
    path2write=savepath,  # Path to save model and logs
    scaler=scaler  # Scaler for mixed precision training
)

print("Training the Model...")

# Run the training process
training_losses, validation_losses, lr_rates = trainer.run_trainer()

# Save the trained model
model_name = 'glomeruli_model.pth'
torch.save(model.state_dict(), os.path.join(savepath, model_name))  # Save model weights
