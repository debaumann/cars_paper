import torch 
import numpy as np
from numpy.typing import NDArray
import os
import sys 
import matplotlib.pyplot as plt
import cv2

hand_path = '/Users/dennisbaumann/cars_paper/data/val/hand_masks/'
obj_path = '/Users/dennisbaumann/cars_paper/data/val/obj_masks/'
idx = 42

hand_mask = np.load(os.path.join(hand_path, f'{idx:03d}.npy'))
obj_mask = np.load(os.path.join(obj_path, f'{idx:03d}.npy'))

print(hand_mask.shape)
print(obj_mask.shape)
fixed_max_value = 255

fig, ax = plt.subplots(3,8,figsize=(12, 8))
for i in range(8):
    ax[0, i].imshow(hand_mask[i], cmap='hot', interpolation='nearest', vmin=0, vmax=fixed_max_value)
    ax[1, i].imshow(obj_mask[i], cmap='hot', interpolation='nearest', vmin=0, vmax=fixed_max_value)
    ax[2, i].imshow(hand_mask[i] + obj_mask[i], cmap='hot', interpolation='nearest', vmin=0, vmax=fixed_max_value)
plt.tight_layout()
plt.show()
