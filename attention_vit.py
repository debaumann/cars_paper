from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wandb
import os
from utils.vit_train_utils import MultiModalDataset
from torch.utils.data import DataLoader








data_root = '/path/to/data'

# Define subject splits (for example, training: S01–S07, validation: S08–S09)
train_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07']
val_subjects = ['S08', 'S09']

# Create dataset instances for training and validation.
train_dataset = MultiModalDataset(data_root, train_subjects)
val_dataset = MultiModalDataset(data_root, val_subjects)

# Wrap the datasets in DataLoaders.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)