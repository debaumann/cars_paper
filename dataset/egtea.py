import numpy as np 
import os
import sys
import time
import glob
import matplotlib.pyplot as plt
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.gaze_io_sample import parse_gtea_gaze



path ='/cluster/scratch/debaumann/egtea/egtea'


def get_split_addresses(path,split_num):

    train_split = path + '/action_annotation/train_split'+str(split_num)+'.txt'
    test_split = path + '/action_annotation/test_split'+str(split_num)+'.txt'

    with open(train_split, 'r') as file:
        train_data = file.readlines()
        train_data= sorted(train_data)
    
    with open(test_split, 'r') as file:
        test_data = file.readlines()
        test_data= sorted(test_data)
    return train_data, test_data

def video_to_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not os.path.exists(video_path):
        raise IOError(f"Video file does not exist: {video_path}")
    
    if not cap.isOpened():
        raise IOError("Could not open video")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.uint8)
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    
    len_frames = len(frames)
    frames_idxs = np.linspace(0, len_frames-1, num=8, dtype=np.uint32)
    sampled_frames = frames[frames_idxs]

    return sampled_frames,frames_idxs

def get_addresses(address):
    split = address.split(' ')
    video_split = split[0].split('-')
    video_path = path + f'/cropped_clips/{video_split[0]}-{video_split[1]}-{video_split[2]}/{split[0]}.mp4'
    gaze = path + f'/gaze_data/gaze_data/{video_split[0]}-{video_split[1]}-{video_split[2]}.txt'
    frame_start = video_split[5]
    frame_end = video_split[6]
    frame_start = int(frame_start[1:])
    frame_end = int(frame_end[1:])
    return video_path,gaze, frame_start, frame_end
def get_data(adress):
    vid1, gaze,f_start,f_end = get_addresses(adress)
    frames, frames_idx = video_to_frames(vid1)


    gaze_data = parse_gtea_gaze(gaze)
    gazes = []
    for frame_idx in frames_idx:
        try:
            gaze = gaze_data[frame_idx]
        except KeyError:
            # Set gaze to center if not available
            gaze = [640,480,1]
        gazes.append(gaze)
    label = adress.split(' ')[1]
    return frames, gazes, label

def create_heatmap_mask(image_shape, gaze_point, sigma=40, kernel_size=21):
    """
    Create a heatmap mask with a Gaussian distribution centered at the gaze point,
    and then apply a Gaussian blur and colormap for visualization.

    Parameters:
        image_shape (tuple): The (height, width) of the image.
        gaze_point (tuple): (x, y) coordinates for the gaze center.
        sigma (float): Standard deviation for the Gaussian.
        kernel_size (int): Kernel size for an additional Gaussian blur.

    Returns:
        heatmap (ndarray): A colored heatmap image (BGR).
        mask (ndarray): A grayscale mask with values in the range [0, 1].
    """
    h, w = image_shape
    x_center = gaze_point[0]*0.5
    y_center = gaze_point[1]*0.5
    gaze_type = gaze_point[2]


    # Create a coordinate grid for the image
    x = np.arange(0, w)
    y = np.arange(0, h)
    xx, yy = np.meshgrid(x, y)

    # Compute a Gaussian distribution centered at the gaze point
    if gaze_type == 1:
        mask = np.exp(-((xx - x_center)**2 + (yy - y_center)**2) / (2 * sigma**2))
        mask = mask / mask.max()  # Normalize to [0, 1]

        # Optionally, apply an additional Gaussian blur to further smooth the mask
        mask_blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

        # Normalize the blurred mask to [0, 255] for colormap application
        mask_blurred_norm = np.uint8(255 * (mask_blurred / mask_blurred.max()))

    # Apply a colormap (e.g., JET) to create a colorful heatmap visualization
    else:
        mask_blurred_norm = np.zeros((h, w), dtype=np.uint8)
    
    return mask_blurred_norm



snums = [1,2,3]
for num in snums:
    train_split,test_split = get_split_addresses(path,num)

    save_labels = f'{path}/labels_split{num}/train'
    save_images = f'{path}/images_split{num}/train'
    os.makedirs(save_images, exist_ok=True)
    save_heatmaps = f'{path}/heatmaps_split{num}/train'
    os.makedirs(save_heatmaps, exist_ok=True)
    os.makedirs(save_labels, exist_ok=True)
    for i in range(len(train_split)):
        frames,gazes,labels = get_data(train_split[i])
        np.save(save_images + f'/{i:05d}.npy', frames)
        np.save(save_labels + f'/{i:05d}.npy', labels)
        np.save(save_heatmaps + f'/{i:05d}.npy', gazes)
    save_labels = f'{path}/labels_split{num}/test'
    save_images = f'{path}/images_split{num}/test'
    os.makedirs(save_images, exist_ok=True)
    save_heatmaps = f'{path}/heatmaps_split{num}/test'
    os.makedirs(save_heatmaps, exist_ok=True)
    os.makedirs(save_labels, exist_ok=True)
    for i in range(len(test_split)):
        frames,gazes,labels = get_data(test_split[i])
        np.save(save_images + f'/{i:05d}.npy', frames)
        np.save(save_labels + f'/{i:05d}.npy', labels)
        np.save(save_heatmaps + f'/{i:05d}.npy', gazes)









