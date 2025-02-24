
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
from utils.vit_train_utils import MultiModalDataset, Cars_Action,preprocess,set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm



data_root = '/cluster/scratch/debaumann/arctic_data'


# Define subject splits (for example, training: S01–S07, validation: S08–S09)
train_subjects = ['S01', 'S02', 'S04', 'S07', 'S08', 'S09', 'S10']
val_subjects = ['S05', 'S06']
seed = 7
set_seed(seed)
train_dataset = MultiModalDataset(data_root, train_subjects)
val_dataset = MultiModalDataset(data_root, val_subjects)

# Wrap the datasets in DataLoaders.
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)


save_batch_dir = '/cluster/home/debaumann/cars_paper/data_check'
os.makedirs(save_batch_dir, exist_ok=True)
epoch = 0
num_epochs = 1
for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}")):
        imgs, hand_heatmap, object_heatmap, labels = batch
        hand_heatmap = hand_heatmap.squeeze(0).squeeze(1)
        object_heatmap = object_heatmap.squeeze(0).squeeze(1)
        
        hand_heatmap = hand_heatmap.squeeze(1)
        object_heatmap = object_heatmap.squeeze(1)
        if batch_idx % 10 == 0:
            for j in range(min(1, imgs.shape[0])):
                    # Row 0: Input image
                print(imgs[0,j].shape)
                img_np = imgs[0,j].permute(1, 2, 0).cpu().numpy()
                
                obj_np =object_heatmap[j].squeeze(0).cpu().numpy()
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(img_np/255)
                plt.title('Input Image')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(obj_np, cmap='hot')
                plt.title('Object Heatmap')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_batch_dir, f"batch_{batch_idx}_img_{j}_combined.png"))
                plt.close()

                    