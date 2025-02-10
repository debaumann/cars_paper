import torch
from manopth.manolayer import ManoLayer
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys 
import os 
import imageio
import cv2
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
    curr_egocam, curr_mano, curr_object, curr_smplx = create_subfolder_name(curr_adress, sid)
    data = [np.load(os.path.join(f'{path_to_data}/poses/{sid}', f), allow_pickle=True) for f in [curr_egocam, curr_mano, curr_object, curr_smplx]]
    data_dicts = []
    for obj in data:
        if isinstance(obj, np.ndarray) and obj.dtype == 'O':
            obj_dict = {key: value for key, value in obj.item().items()}
            data_dicts.append(obj_dict)
        else:
            data_dicts.append(obj)
    images_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    images = [cv2.imread(os.path.join(image_dir, f)) for f in images_files]
    frames_arr = np.linspace(curr_start, curr_end, 8, dtype=np.int32)
    return data_dicts, images,frames_arr

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


    # Read the address list from the labels path
    labels_path = path_to_data + f'/labels/{sid}.csv'
    labels_df = pd.read_csv(labels_path)
    adress_book = labels_df['address'].values
    start_frames = labels_df['start_frame'].values
    end_frames = labels_df['end_frame'].values
    numeric_labels = labels_df['numeric_label'].values

    save_dir_hands = f'{path_to_data}/hand_heatmaps/{mode}/{sid}/'
    save_dir_obj = f'{path_to_data}/object_heatmaps/{mode}/{sid}/'
    os.makedirs(save_dir_hands, exist_ok=True)
    os.makedirs(save_dir_obj, exist_ok=True)

    # Generate random shape parameters

    for i in range(len(adress_book)):
        
        
        data_dicts,images,frames_arr = create_dicts(adress_book,path_to_data,start_frames,end_frames, sid,i)
        mano_dict = data_dicts[1]
        camera_dict = data_dicts[0]
        object_arr = np.array(data_dicts[2])
        mano_left= mano_dict['left']
        mano_right = mano_dict['right']
        distortions = np.array(camera_dict['dist8'])
        R0= np.array(camera_dict['R0'])
        T0 = np.array(camera_dict['T0'])*1000



        orig_mesh_faces_top, orig_mesh_faces_bottom, orig_mesh_verts_top, orig_mesh_verts_bottom = get_mesh_vertices()
        

        # Example usage
        hand_heatmaps = []
        obj_heatmaps = []


        for idx in frames_arr:
            l_bool = True
            r_bool = True
            top_bool = True
            bot_bool = True

            print(f'Processing frame {idx}')
            left_rot = mano_left['rot'][idx]
            left_trans= mano_left['trans'][idx]
            left_pose = mano_left['pose'][idx]
            left_pose = np.concatenate((left_rot, left_pose), axis=0)
            left_shape = mano_left['shape']
            random_pose = torch.tensor(np.array(left_pose, np.float32)).unsqueeze(0)  # Add batch dimension
            random_tran = torch.tensor(np.array(left_trans, np.float32)).unsqueeze(0)  # Add batch dimension
            random_shape = torch.tensor(np.array(left_shape, np.float32)).unsqueeze(0)  # Add batch dimension

            mano_left_points = mano_layer_left(random_pose, random_shape)
            l_hand_verts_scaled = mano_left_points[0] / 1000.0 + random_tran
            l_hand_verts_scaled = l_hand_verts_scaled.detach().numpy()[0]
            

            cam_r = camera_dict['R_k_cam_np'][idx]
            cam_t = camera_dict['T_k_cam_np'][idx]
            intrinsics = np.array(camera_dict['intrinsics'])

            # Project the 3D hand vertices into the 2D camera image
            l_hand_verts_hom = np.concatenate([l_hand_verts_scaled, np.ones((l_hand_verts_scaled.shape[0], 1))], axis=1).T
            cam_extrinsics = np.hstack((cam_r, cam_t.reshape(-1, 1)))
            
            cam_extrinsics = np.vstack((cam_extrinsics, [0, 0, 0, 1]))
            
            cam_intrinsics = intrinsics


            l_hand_verts_cam = cam_extrinsics @ l_hand_verts_hom
            dist8 = distortions
            l_hand_verts_distort = distort_pts3d_all(l_hand_verts_cam.T, dist8)

            l_hand_verts_img = cam_intrinsics @ l_hand_verts_distort.T[:3, :]
            l_hand_verts_img = l_hand_verts_img[:2, :] / l_hand_verts_img[2, :]


            ##### mano projection for right hand
            right_rot = mano_right['rot'][idx]
            right_trans= mano_right['trans'][idx]
            right_pose = mano_right['pose'][idx]
            right_pose = np.concatenate((right_rot, right_pose), axis=0)
            right_shape = mano_right['shape']
            random_pose = torch.tensor(np.array(right_pose, np.float32)).unsqueeze(0)  # Add batch dimension
            random_tran = torch.tensor(np.array(right_trans, np.float32)).unsqueeze(0)  # Add batch dimension
            random_shape = torch.tensor(np.array(right_shape, np.float32)).unsqueeze(0)  #
            mano_right_points = mano_layer_right(random_pose, random_shape)
            r_hand_verts_scaled = mano_right_points[0] / 1000.0 + random_tran
            r_hand_verts_scaled = r_hand_verts_scaled.detach().numpy()[0]
            r_hand_verts_hom = np.concatenate([r_hand_verts_scaled, np.ones((r_hand_verts_scaled.shape[0], 1))], axis=1).T
            r_hand_verts_cam = cam_extrinsics @ r_hand_verts_hom
            r_hand_verts_distort = distort_pts3d_all(r_hand_verts_cam.T, dist8)
            r_hand_verts_img = cam_intrinsics @ r_hand_verts_distort.T[:3, :]
            r_hand_verts_img = r_hand_verts_img[:2, :] / r_hand_verts_img[2, :]



            
            # object mesh
            obj_array = object_arr[idx]
            articulation_angle = np.rad2deg(-obj_array[0])
            rot_angle_axis = obj_array[1:4]
            obj_trans = obj_array[4:7]/1000
            obj_rot = R.from_rotvec(rot_angle_axis)
            obj_rot_matrix = obj_rot.as_matrix()
            obj_t = np.hstack((obj_rot_matrix,  (obj_trans).reshape(-1, 1)))
            canonical_T = np.hstack((R0, T0.reshape(-1, 1)))
            canonical_T = np.vstack((canonical_T, [0, 0, 0, 1]))
            obj_transformation = np.vstack((obj_t, [0, 0, 0, 1]))
            # obj_transformation = obj_transformation@canonical_T
            # Apply rotation and translation to the object mesh vertices
            mesh_faces_top = np.copy(orig_mesh_faces_top)
            mesh_faces_bottom = np.copy(orig_mesh_faces_bottom)
            mesh_verts_top = np.copy(orig_mesh_verts_top)
            mesh_verts_bottom = np.copy(orig_mesh_verts_bottom)
            mesh_faces_top, mesh_verts_top = articulate_object(articulation_angle, mesh_faces_top, mesh_faces_bottom, mesh_verts_top, mesh_verts_bottom)
            mesh_verts_top /= 1000
            mesh_verts_bottom /= 1000
            obj_verts_top_hom = np.concatenate([mesh_verts_top, np.ones((mesh_verts_top.shape[0], 1))], axis=1).T
            obj_verts_top_world  = obj_transformation@obj_verts_top_hom
            
            obj_verts_top_cam = cam_extrinsics @ obj_verts_top_world
            # obj_verts_cam = obj_verts_cam
            

            obj_verts_top_distort = distort_pts3d_all(obj_verts_top_cam[:3,:].T, dist8)
            obj_verts_top_img = cam_intrinsics @ obj_verts_top_distort.T
            obj_verts_top_img = obj_verts_top_img[:2, :] / obj_verts_top_img[2, :]
            
            obj_verts_bottom_hom = np.concatenate([mesh_verts_bottom, np.ones((mesh_verts_bottom.shape[0], 1))], axis=1).T
            obj_verts_bottom_world  = obj_transformation@obj_verts_bottom_hom
            obj_verts_bottom_cam = cam_extrinsics @ obj_verts_bottom_world
            obj_verts_bottom_distort = distort_pts3d_all(obj_verts_bottom_cam[:3,:].T, dist8)
            obj_verts_bottom_img = cam_intrinsics @ obj_verts_bottom_distort.T
            obj_verts_bottom_img = obj_verts_bottom_img[:2, :] / obj_verts_bottom_img[2, :]
            obj_verts_img = np.hstack((obj_verts_top_img, obj_verts_bottom_img))
            # Plot the projected points on the image

            #####check out of image for distort#######
            l_min_y_idx = np.argmin(np.abs(l_hand_verts_cam[1,:]))
            l_angle = np.arctan2(l_hand_verts_cam[1, l_min_y_idx], l_hand_verts_cam[2, l_min_y_idx])
            if np.abs(np.rad2deg(l_angle)) > 19:
                l_bool = False
            
            r_min_y_idx = np.argmin(np.abs(r_hand_verts_cam[1,:]))
            r_angle = np.arctan2(r_hand_verts_cam[1, r_min_y_idx], r_hand_verts_cam[2, r_min_y_idx])
            if np.abs(np.rad2deg(r_angle)) > 19:
                r_bool = False
            
            top_min_y_idx = np.argmin(np.abs(obj_verts_top_cam[1,:]))
            top_angle = np.arctan2(obj_verts_top_cam[1, top_min_y_idx], obj_verts_top_cam[2, top_min_y_idx])
            if np.abs(np.rad2deg(top_angle)) > 19:
                top_bool = False
            
            bot_min_y_idx = np.argmin(np.abs(obj_verts_bottom_cam[1,:]))
            bot_angle = np.arctan2(obj_verts_bottom_cam[1, bot_min_y_idx], obj_verts_bottom_cam[2, bot_min_y_idx])
            if np.abs(np.rad2deg(bot_angle)) > 19:
                bot_bool = False
            ############ hand heatmap stuff ##############
            if hand_heatmap:
                mask = np.zeros((2000,2800, 3), dtype=np.uint8)
                l_mask = np.zeros((600,840, 3), dtype=np.uint8)
                r_mask = np.zeros((600,840, 3), dtype=np.uint8)
                l_faces = mano_layer_left.th_faces.numpy()
                r_faces = mano_layer_right.th_faces.numpy()

                l_verts = l_hand_verts_cam[:3,:].T
                r_verts = r_hand_verts_cam[:3,:].T
                print(l_verts.shape, r_verts.shape)
                obj_verts = np.hstack((obj_verts_top_cam[:3,:], obj_verts_bottom_cam[:3,:]))
                print(obj_verts.shape)
                l_depth_buffer = np.full((600,840), 0)
                r_depth_buffer = np.full((600,840), 0)
                l_points_dist= []
                for j in range(l_verts.shape[0]):
                    p = l_verts[j]
                    dist = compute_dist_to_mesh(p, obj_verts.T)
                    #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
                    l_points_dist.append(dist)
                r_points_dist= []
                for j in range(r_verts.shape[0]):
                    p = r_verts[j]
                    dist = compute_dist_to_mesh(p, obj_verts.T)
                    #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
                    r_points_dist.append(dist)
                l_verts_img = l_hand_verts_img.T
                r_verts_img = r_hand_verts_img.T
                if l_bool:
                    for face in l_faces:
                            pts = []
                            dists = []
                            for elem in face:
                                pts.append(l_verts_img[elem])
                                dists.append(l_points_dist[elem])
                            pts = np.array(pts, dtype=np.int32)
                            # Scale the points to the new mask size:
                            pts_scaled = (pts * 0.3).astype(np.int32)
                            pts_scaled = pts_scaled.reshape((-1, 1, 2))
                            mean_dist = np.mean(dists)
                            face_depth = get_intensity(mean_dist)*255
                            # Create a mask for the face
                            mask = np.zeros((600,840), dtype=np.uint8)
                            cv2.fillPoly(mask, [pts_scaled], color=1)
                            mask_indices = np.where(mask == 1)
                            for y, x in zip(*mask_indices):
                                if face_depth > l_depth_buffer[y, x]:
                                    l_depth_buffer[y, x] = face_depth
                                    l_mask[y, x] = (face_depth, face_depth, face_depth)
                if r_bool:
                    for face in r_faces:
                        pts = []
                        dists = []
                        for elem in face:
                            pts.append(r_verts_img[elem])
                            dists.append(r_points_dist[elem])
                        pts = np.array(pts, dtype=np.int32)
                            # Scale the points to the new mask size:
                        pts_scaled = (pts * 0.3).astype(np.int32)
                        pts_scaled = pts_scaled.reshape((-1, 1, 2))
                        mean_dist = np.mean(dists)
                        face_depth = get_intensity(mean_dist)*255
                        # Create a mask for the face
                        mask = np.zeros((600,840), dtype=np.uint8)
                        cv2.fillPoly(mask, [pts_scaled], color=1)
                        mask_indices = np.where(mask == 1)
                        for y, x in zip(*mask_indices):
                            if face_depth > r_depth_buffer[y, x]:
                                r_depth_buffer[y, x] = face_depth
                                r_mask[y, x] = (face_depth, face_depth, face_depth)

                r_save = r_mask[:,:,0]
                l_save = l_mask[:,:,0]

                combined_save = np.maximum(r_save, l_save).astype(np.uint8)
                # resized_img = cv2.resize(combined_save, (0, 0), fx=0.3, fy=0.3).astype(np.uint8)
                hand_heatmaps.append(combined_save)
                # crop_h = 600
                # crop_w = 840
                # img = img[i_h//2-crop_h//2:i_h//2+crop_h//2, i_w//2-crop_w//2:i_w//2+crop_w//2, :]
                # print(f'Max image position: ({np.max(l_hand_verts_img[0, :])}, {np.max(l_hand_verts_img[1, :])})')
                # print(f'Min image position: ({np.min(l_hand_verts_img[0, :])}, {np.min(l_hand_verts_img[1, :])})')
                # resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
                # # plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
                # overlay_img = cv2.addWeighted(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB), 0.5, resized_img, 0.5, 0)
                # plt.imsave(save_dir_hands+ f'{idx:03d}.png', overlay_img)
            
            ################ object heatmap stuff###############
            if object_heatmap:
                top_mask = np.zeros((600,840, 3), dtype=np.uint8)
                bot_mask = np.zeros((600,840, 3), dtype=np.uint8)
                top_faces = mesh_faces_top
                bot_faces = mesh_faces_bottom

                top_verts = obj_verts_top_cam[:3,:].T
                bot_verts = obj_verts_bottom_cam[:3,:].T

                hand_verts = np.hstack((l_hand_verts_cam[:3,:], r_hand_verts_cam[:3,:]))
                
                top_depth_buffer = np.full((600,840), 0)
                bot_depth_buffer = np.full((600,840), 0)
                top_points_dist= []
                for k in range(top_verts.shape[0]):
                    p = top_verts[k]
                    dist = compute_dist_to_mesh(p, hand_verts.T)
                    #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
                    top_points_dist.append(dist)
                bot_points_dist= []
                for k in range(bot_verts.shape[0]):
                    p = bot_verts[k]
                    dist = compute_dist_to_mesh(p, hand_verts.T)
                    #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
                    bot_points_dist.append(dist)
                top_verts_img = obj_verts_top_img.T
                bot_verts_img = obj_verts_bottom_img.T
                if top_bool:
                    for face in mesh_faces_top:
                            pts = []
                            dists = []
                            for elem in face:
                                pts.append(top_verts_img[elem])
                                dists.append(top_points_dist[elem])
                            pts = np.array(pts, dtype=np.int32)
                            # Scale the points to the new mask size:
                            pts_scaled = (pts * 0.3).astype(np.int32)
                            pts_scaled = pts_scaled.reshape((-1, 1, 2))
                            mean_dist = np.mean(dists)
                            face_depth = get_intensity(mean_dist)*255
                            # Create a mask for the face
                            mask = np.zeros((600,840), dtype=np.uint8)
                            cv2.fillPoly(mask, [pts_scaled], color=1)
                            mask_indices = np.where(mask == 1)
                            for y, x in zip(*mask_indices):
                                if face_depth > top_depth_buffer[y, x]:
                                    top_depth_buffer[y, x] = face_depth
                                    top_mask[y, x] = (face_depth, face_depth, face_depth)
                if bot_bool:
                    for face in mesh_faces_bottom:
                        pts = []
                        dists = []
                        for elem in face:
                            pts.append(bot_verts_img[elem])
                            dists.append(bot_points_dist[elem])
                        pts = np.array(pts, dtype=np.int32)
                        # Scale the points to the new mask size:
                        pts_scaled = (pts * 0.3).astype(np.int32)
                        pts_scaled = pts_scaled.reshape((-1, 1, 2))
                        mean_dist = np.mean(dists)
                        face_depth = get_intensity(mean_dist)*255
                        # Create a mask for the face
                        mask = np.zeros((600,840), dtype=np.uint8)
                        cv2.fillPoly(mask, [pts_scaled], color=1)
                        mask_indices = np.where(mask == 1)
                        for y, x in zip(*mask_indices):
                            if face_depth > bot_depth_buffer[y, x]:
                                bot_depth_buffer[y, x] = face_depth
                                bot_mask[y, x] = (face_depth, face_depth, face_depth)

                top_save = top_mask[:,:,0]
                bot_save = bot_mask[:,:,0]

                combined_save = np.maximum(top_save, bot_save).astype(np.uint8)
                
                

                # resized_img = cv2.resize(combined_save, (0, 0), fx=0.3, fy=0.3).astype(np.uint8)
                # crop_h = 600
                # crop_w = 840
                # img = img[i_h//2-crop_h//2:i_h//2+crop_h//2, i_w//2-crop_w//2:i_w//2+crop_w//2, :]
                # print(f'Max image position: ({np.max(l_hand_verts_img[0, :])}, {np.max(l_hand_verts_img[1, :])})')
                # print(f'Min image position: ({np.min(l_hand_verts_img[0, :])}, {np.min(l_hand_verts_img[1, :])})')
                
                obj_heatmaps.append(combined_save)



        



            
            
        
            

        #     gif.append(output_image)
        

        
        np.save(save_dir_hands + f'{i:04d}.npy', np.array(hand_heatmaps))
        np.save(save_dir_obj + f'{i:04d}.npy', np.array(obj_heatmaps))
        print(f'Finished {i:04d}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--slurm_id", type=int, required=True, help="Start index for the loop")
    args = parser.parse_args()

    main(args.slurm_id)
    
