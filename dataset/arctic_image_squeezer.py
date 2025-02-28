import torch
from manopth.manolayer import ManoLayer
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys 
import os 
import imageio
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.heatmap_projection_utils import project_points, project_joint_points, dist_to_bbox, get_intensity,compute_dist_to_mesh, plot_two_hands
from utils.obj_mesh_arctic import get_mesh_vertices
from scipy.spatial.transform import Rotation as R
import pandas as pd
import argparse

hand_heatmap = True
object_heatmap = True

batch_size = 1
# Select number of principal components for pose space
ncomps = 45


mode = 'train'
#gif_path = '/Users/dennisbaumann/cars_paper/mano_test_output_val/'
#os.makedirs(gif_path, exist_ok=True)
# Initialize MANO layer
mano_layer_left = ManoLayer(mano_root='/cluster/home/debaumann/manopth/mano/models', use_pca=False, flat_hand_mean=False, ncomps=ncomps, side='left')
mano_layer_right= ManoLayer(mano_root='/cluster/home/debaumann/manopth/mano/models', use_pca=False,flat_hand_mean=False, ncomps=ncomps, side='right')



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
    image_dir = f'{path_to_data}/images/{curr_adress}'
    curr_start = start_frames[idx]
    curr_end = end_frames[idx]
    images_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    images = [cv2.imread(os.path.join(image_dir, f)) for f in images_files]
    frames_arr = np.linspace(curr_start, curr_end, 8, dtype=np.int32)
    return images,frames_arr

def articulate_object(angle, mesh_faces_top, mesh_faces_bottom, mesh_verts_top, mesh_verts_bottom):
        # Rotate the object mesh
        r = R.from_euler('z', angle, degrees=True)
        mesh_verts_top = r.apply(mesh_verts_top)
        return mesh_faces_top, mesh_verts_top

def distort_pts3d_all(pts_cam, dist_coeffs):
    """
    Apply distortion to 3D points in the camera coordinate system.

    Parameters:
    - pts_cam: numpy array of shape (N, M, 3), where N is the batch size and M is the number of points.
    - dist_coeffs: numpy array of distortion coefficients.

    Returns:
    - cam_pts_dist: numpy array of distorted 3D points.
    """
    pts_cam = pts_cam.astype(np.float64)
    z = pts_cam[ :, 2]

    z_inv = 1 / z

    x1 = pts_cam[ :, 0] * z_inv
    y1 = pts_cam[ :, 1] * z_inv

    # Precalculations
    x1_2 = x1 * x1
    y1_2 = y1 * y1
    x1_y1 = x1 * y1
    r2 = x1_2 + y1_2
    r4 = r2 * r2
    r6 = r4 * r2

    r_dist = (1 + dist_coeffs[0] * r2 + dist_coeffs[1] * r4 + dist_coeffs[4] * r6) / (
        1 + dist_coeffs[5] * r2 + dist_coeffs[6] * r4 + dist_coeffs[7] * r6
    )

    # Full (rational + tangential) distortion
    x2 = x1 * r_dist + 2 * dist_coeffs[2] * x1_y1 + dist_coeffs[3] * (r2 + 2 * x1_2)
    y2 = y1 * r_dist + 2 * dist_coeffs[3] * x1_y1 + dist_coeffs[2] * (r2 + 2 * y1_2)

    # Denormalize for projection (which is a linear operation)
    cam_pts_dist = np.stack([x2 * z, y2 * z, z], axis=1).astype(np.float32)
    return cam_pts_dist

def main(slurm_id):
    sid = f'S{slurm_id:02d}'
    path_to_data = '/cluster/home/debaumann/cars_paper/arctic_data'
    path_to_scratch = '/cluster/scratch/debaumann/arctic_data'


    # Read the address list from the labels path
    labels_path = path_to_data + f'/labels/{sid}.csv'
    labels_df = pd.read_csv(labels_path)
    adress_book = labels_df['address'].values
    start_frames = labels_df['start_frame'].values
    end_frames = labels_df['end_frame'].values
    numeric_labels = labels_df['numeric_label'].values

    save_dir_image= f'{path_to_scratch}/images/{mode}/{sid}/'
    os.makedirs(save_dir_image, exist_ok=True)

    # Generate random shape parameters

    for i in range(len(adress_book)):
        
        
        images,frames_arr = create_dicts(adress_book,path_to_data,start_frames,end_frames, sid,i)
        
        frames_array = []


        for idx in frames_arr:
            curr_frame = images[idx]
            curr_frame= cv2.resize(curr_frame, (720, 720), interpolation=cv2.INTER_AREA)
            frames_array.append(curr_frame.astype(np.uint8))
        
            
        
        np.save(save_dir_image + f'{i:04d}.npy', np.array(frames_array).astype(np.uint8))
        print(f'Finished {i:04d}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--slurm_id", type=int, required=True, help="Start index for the loop")
    args = parser.parse_args()

    main(args.slurm_id)
    
