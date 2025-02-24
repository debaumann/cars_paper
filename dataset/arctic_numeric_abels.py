import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys 
import os 
import imageio
import cv2
import pandas as pd

hdd_path = '/Volumes/cars_debaumann/arctic_data/'

def isolate_folder_name(address, sid):
    parts = address.split('/')
    if len(parts) > 2 and parts[0] == sid:
        return parts[1]
    return None
def create_subfolder_name(address, sid):
    address = isolate_folder_name(address, sid)
    base_name = address.split('/')[-1]
    return [f'{base_name}.egocam.dist.npy', f'{base_name}.mano.npy', f'{base_name}.object.npy', f'{base_name}.smplx.npy']



def create_dicts(adress_book,path_to_data,start_frames,end_frames, sid, idx):
    curr_adress = adress_book[idx]
    image_dir = f'/Users/dennisbaumann/cars_paper/data/arctic_data/{curr_adress}'
    curr_start = start_frames[idx]
    curr_end = end_frames[idx]
    curr_egocam, curr_mano, curr_object, curr_smplx = create_subfolder_name(curr_adress, sid)
    data = [np.load(os.path.join(f'{path_to_data}/poses/{sid}', f), allow_pickle=True) for f in [curr_egocam, curr_mano, curr_object, curr_smplx]]
    data_dicts = []
    for obj in data:
        if isinstance(obj, np.ndarray) and obj.dtype == 'O':
            obj_dict = {key: value for key, value in obj.item().items()}
            data_dicts.append(obj_dict)
        else:
            data_dicts.append(obj)
    # images_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    # images = [cv2.imread(os.path.join(image_dir, f)) for f in images_files]
    images = []
    frames_arr = np.linspace(curr_start, curr_end, 8, dtype=np.int32)
    return data_dicts, images,frames_arr

for k in range(1,11):
    sid = f'S{k:02d}'
    


    # Read the address list from the labels path
    labels_path = hdd_path + f'/labels/{sid}.csv'
    labels_df = pd.read_csv(labels_path)
    adress_book = labels_df['address'].values
    start_frames = labels_df['start_frame'].values
    end_frames = labels_df['end_frame'].values
    numeric_labels = labels_df['numeric_label'].values
    

    source_dir_hands = f'{hdd_path}/hand_heatmaps/train/{sid}/'
    source_dir_obj = f'{hdd_path}/object_heatmaps/train/{sid}/'
    save_dir_hands = f'{hdd_path}/hand_heatmaps_squeezed/train/{sid}/'
    save_dir_obj = f'{hdd_path}/object_heatmaps_squeezed/train/{sid}/'
    save_labels_dir = f'{hdd_path}/action_labels/train/{sid}/'
    os.makedirs(save_labels_dir, exist_ok=True)


    for i in range(len(adress_book)):
            
            
        np.save(f'{save_labels_dir}{i:04d}.npy', numeric_labels[i])
        