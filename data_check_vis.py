import torch 
import numpy as np
from numpy.typing import NDArray
import os
import sys 
import matplotlib.pyplot as plt
import cv2
mode= 'val'
hand_path = f'/Users/dennisbaumann/cars_paper/data/{mode}/hand_masks/'
obj_path = f'/Users/dennisbaumann/cars_paper/data/{mode}/obj_masks/'
im_path = f'/Users/dennisbaumann/cars_paper/data/{mode}/seq_8_{mode}/'
idx = 0


hand_mask = np.load(os.path.join(hand_path, f'{idx:03d}.npy'))
obj_mask = np.load(os.path.join(obj_path, f'{idx:03d}.npy'))
images = np.load(os.path.join(im_path, f'{idx:03d}.npy'))

print(hand_mask.shape)
print(obj_mask.shape)
fixed_max_value = 255
hand_heatmap =torch.from_numpy(hand_mask/255.0).float()
obj_heatmap = torch.from_numpy(obj_mask/255.0).float()
mse = torch.nn.MSELoss()
loss = mse(hand_heatmap, obj_heatmap)
print(loss)
other_loss = mse(hand_heatmap, hand_heatmap)
print(other_loss)
hand_np = hand_heatmap.numpy()


im = np.array(images[0])/255
hand_obj = hand_mask[0] + obj_mask[0]
hand_obj_im = np.zeros((hand_obj.shape[0], hand_obj.shape[1], 3))
hand_obj_im[:,:,0] = hand_obj/255
print(np.max(hand_mask[0]))
im = im.astype(np.float32)
hand_obj_im = hand_obj_im.astype(np.float32)

# Overlay the hand_obj_im on top of im
alpha = 0.5  # Transparency factor
comb_image = cv2.addWeighted(im, 1.0, hand_obj_im, 1.0, 0)

plt.imshow(comb_image)
plt.show()
fig, ax = plt.subplots(4,8,figsize=(10, 3))
for i in range(8):
    im = np.array(images[i])/255
    hand_obj = hand_mask[i] + obj_mask[i]
    hand_obj_im = np.zeros((hand_obj.shape[0], hand_obj.shape[1], 3))
    hand_obj_im[:,:,0] = hand_obj/255
    print(np.max(hand_mask[i]))
    im = im.astype(np.float32)
    hand_obj_im = hand_obj_im.astype(np.float32)

    # Overlay the hand_obj_im on top of im
    alpha = 0.5  # Transparency factor
    comb_image = cv2.addWeighted(im, 1.0, hand_obj_im, 1.0, 0)

    ax[0, i].imshow(hand_mask[i], cmap='hot', interpolation='nearest', vmin=0, vmax=fixed_max_value)
    ax[1, i].imshow(obj_mask[i], cmap='hot', interpolation='nearest', vmin=0, vmax=fixed_max_value)
    ax[2, i].imshow(hand_mask[i] + obj_mask[i], cmap='hot', interpolation='nearest', vmin=0, vmax=fixed_max_value)
    ax[3, i].imshow(comb_image)
plt.imshow(comb_image)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(8, 1, figsize=(5,10))
for i in range(8):
    im = np.array(images[i])/255
    hand_obj = hand_mask[i] + obj_mask[i]
    hand_obj_im = np.zeros((hand_obj.shape[0], hand_obj.shape[1], 3))
    hand_obj_im[:,:,0] = hand_obj/255
    im = im.astype(np.float32)
    hand_obj_im = hand_obj_im.astype(np.float32)
    comb_image = cv2.addWeighted(im, 1.0, hand_obj_im, 1.0, 0)
    ax[i].imshow(comb_image)
    ax[i].axis('off')
plt.tight_layout()
plt.show()
