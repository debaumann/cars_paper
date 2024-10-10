import os 
import sys
import pandas as pd
import numpy as np  
from tqdm import tqdm
from typing import List
from numpy.typing import NDArray
import cv2 as cv

path_to_label = 'D:/action_test.txt'
root_data = 'D:/'

n_frames_per_seq : int = 8

#mode = 'train'
#sample_root = root_data + f'mano_{n_frames_per_seq}_{mode}/'

df = pd.read_csv(path_to_label, delimiter=' ')
print(df.head())    
ids : List[int] = df['id'].to_list()
paths : List[str] = df['path'].to_list()
#action_labels : List[int] = df['action_label'].to_list()
start_acts : List[int] = df['start_act'].to_list()
end_acts : List[int] = df['end_act'].to_list()
start_frames : List[int] = df['start_frame'].to_list()
end_frames : List[int] = df['end_frame'].to_list()

print("Number of action sequences:", len(ids))


#os.makedirs(sample_root, exist_ok=True) # create the destination directory

sample_ids : List[NDArray] = []
# sample the frames for each action
for i in tqdm(range(len(ids))):
    start_act = start_acts[i]
    end_act = end_acts[i]
    n_frames_total = end_act-start_act+1 # total number of frames in the action sequence
    assert n_frames_total >= n_frames_per_seq, \
        f"Requested {n_frames_per_seq} samples, but action (id {ids[i]}) has only {n_frames_total} frames"

    seq_ids = np.linspace(start_acts[i], end_acts[i], n_frames_per_seq, dtype=int)
    sample_ids.append(seq_ids)

sample_ids = np.array(sample_ids)
#np.save(sample_root + 'sample_ids.npy', sample_ids)
print("Sampling IDs shape:", sample_ids.shape) # sanity check the shape of the sampled ids
print("Last sampled sequence ids:", seq_ids) # sanity check the sampling from the last action

manos : list[NDArray] = []
cam_pose : list[NDArray] = []
obj_pose : list[NDArray] = []


for i,id in tqdm(enumerate(sample_ids)):
    path = paths[i]
    manos_list_curr = []
    cam_pose_list_curr = []
    obj_pose_list_curr = []
    rgb_img_list_curr = []
    for j in id:

        path_curr = 'D:/' + path + f'/cam4/hand_pose_mano/{j:06d}.txt'
        cam_pose_curr = np.loadtxt('D:/' + path + f'/cam4/cam_pose/{j:06d}.txt')
        obj_pose_curr = np.loadtxt('D:/' + path + f'/cam4/obj_pose/{j:06d}.txt')
        rgb_img_curr = cv.imread('D:/' + path + f'/cam4/rgb/{j:06d}.png')
        rgb_img_curr = cv.resize(rgb_img_curr, dsize=None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        rgb_img_curr = cv.cvtColor(rgb_img_curr, cv.COLOR_BGR2RGB)
        
        mano_curr  = np.loadtxt(path_curr)
        cam_pose_list_curr.append(np.array(cam_pose_curr))
        manos_list_curr.append(np.array(mano_curr))
        obj_pose_list_curr.append(np.array(obj_pose_curr))
        rgb_img_list_curr.append(np.array(rgb_img_curr, dtype=np.float32))
    
    rgb_img_list_curr = np.stack(rgb_img_list_curr, axis=0)
    
    np.save(f'D:/DEV/CARS_PAPER/Dataset_test/RGB_8_360_640/{i:03d}.npy', rgb_img_list_curr)
    np.save(f'D:/DEV/CARS_PAPER/Dataset_test/MANOS_8/{i:03d}.npy', np.array(manos_list_curr))
    np.save(f'D:/DEV/CARS_PAPER/Dataset_test/CAM_POSE_8/{i:03d}.npy',np.array(cam_pose_list_curr))
    np.save(f'D:/DEV/CARS_PAPER/Dataset_test/OBJ_POSE_8/{i:03d}.npy', np.array(obj_pose_list_curr))

