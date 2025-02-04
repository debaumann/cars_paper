import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys 
import os 
import imageio
import cv2
from utils.heatmap_projection_utils import project_points, project_joint_points, dist_to_bbox, get_intensity,compute_dist_to_mesh, plot_two_hands
from utils.obj_mesh_example import get_mesh_vertices
from scipy.spatial.transform import Rotation as R

print('starting')
#camera params 
fx = 311.76914536239792
fy = 311.76914536239792
u_0 = 319.5
v_0 = 179.5
intrinsics = np.array([[fx, 0, u_0], [0, fy, v_0], [0, 0, 1]])

batch_size = 1
# Select number of principal components for pose space
ncomps = 45


mode = 'train'
gif_path = '/Users/dennisbaumann/cars_paper/mano_test_output_val/'
os.makedirs(gif_path, exist_ok=True)
# Initialize MANO layer
mano_layer_left = ManoLayer(mano_root='/Users/dennisbaumann/manopth/mano/models', use_pca=False, flat_hand_mean=True, ncomps=ncomps, side='left')
mano_layer_right= ManoLayer(mano_root='/Users/dennisbaumann/manopth/mano/models', use_pca=False,flat_hand_mean=True, ncomps=ncomps, side='right')
manos_data_path = f'/Users/dennisbaumann/cars_paper/data/{mode}/mano_8_{mode}/'

mano_pred_path = '/Users/dennisbaumann/cars_paper/data/mano_train/'
cam_data_path = f'/Users/dennisbaumann/cars_paper/data/{mode}/cam_8_{mode}/'
obj_data_path = f'/Users/dennisbaumann/cars_paper/data/{mode}/obj_8_{mode}/'
#joint_data_path = '/Users/dennisbaumann/cars_paper/data/val/joint_8_val/'
obj_rt_data_path = f'/Users/dennisbaumann/cars_paper/data/{mode}/obj_rt_8_{mode}/'
print('loaded data', obj_rt_data_path)
idxs = np.arange(0,569) #0-122 for val, 0-569 for train 
# Generate random shape parameters
save_dir_hands = f'/Users/dennisbaumann/cars_paper/data/{mode}/hand_masks/'
save_dir_obj = f'/Users/dennisbaumann/cars_paper/data/{mode}/obj_masks/'
os.makedirs(save_dir_hands, exist_ok=True)
os.makedirs(save_dir_obj, exist_ok=True)
for idx in idxs:
    test_data = np.load(os.path.join(manos_data_path, f'{idx:03d}.npy'))
    test_pose = np.load(os.path.join(cam_data_path, f'{idx:03d}.npy'))
    test_obj = np.load(os.path.join(obj_data_path, f'{idx:03d}.npy'))
    #test_joints = np.load(os.path.join(joint_data_path, f'{idx:03d}.npy'))
    test_obj_rt = np.load(os.path.join(obj_rt_data_path, f'{idx:03d}.npy'))
    test_obj_rt = np.delete(test_obj_rt, 0, axis=1)

    #pred_pose = np.load(os.path.join(mano_pred_path, f'{idx:03d}.npy'))
    test_image = np.load(f'/Users/dennisbaumann/cars_paper/data/{mode}/seq_8_{mode}/{idx:03d}.npy')
    mesh_points, mesh_verts, mesh_faces= get_mesh_vertices(idx,mode)
    
    
    gif = []
    pose_losses = []
    points_losses = []
    joints_losses = []
    hand_masks = []
    obj_masks = []

    ############start of sequence ################
    for k in range(8):
        test_im = test_image[k].astype(np.uint8)
        load_hand = test_data[k]
        hand_pose = {}
        hand_pose["left_pose"] = [load_hand[4:52]]
        hand_pose["left_tran"] = [load_hand[1:4]]
        hand_pose["left_shape"] = [load_hand[52:62]]

        hand_pose["right_pose"] = [load_hand[66:114]]
        hand_pose["right_tran"] = [load_hand[63:66]]
        hand_pose["right_shape"] = [load_hand[114:124]]

        random_pose = torch.tensor(np.array(hand_pose["left_pose"],np.float32))
        random_tran = torch.tensor(np.array(hand_pose["left_tran"],np.float32))
        random_shape = torch.tensor(np.array(hand_pose["left_shape"],np.float32))


        mano_left_points = mano_layer_left(random_pose, random_shape)
        l_hand_verts_scaled = mano_left_points[0] / 1000.0 + random_tran
        l_hand_verts_scaled = l_hand_verts_scaled.detach().numpy()[0]

        # ones = np.ones((l_hand_verts_scaled.shape[0], 1))
        # l_hand_verts_homogeneous = np.hstack((l_hand_verts_scaled, ones))  # Shape: (N, 4)

        # # Remove the fourth column to make it Nx3
        # l_hand_verts_homogeneous = l_hand_verts_homogeneous[:, :3]

        # # Transpose to get 3xN for matrix multiplication
        # l_hand_verts_homogeneous = l_hand_verts_homogeneous.T  # Shape: (3, N)

        # # Project the points
        # l_image_points_homogeneous = intrinsics @ l_hand_verts_homogeneous  # Shape: (3, N)

        # # Normalize by the third row to get pixel coordinates
        # l_image_points = l_image_points_homogeneous[:2, :] / l_image_points_homogeneous[2, :]

        # # Transpose to get Nx2 array
        # l_points_2d = l_image_points.T 


        random_pose = torch.tensor(np.array(hand_pose["right_pose"],np.float32))
        random_tran = torch.tensor(np.array(hand_pose["right_tran"],np.float32))
        random_shape = torch.tensor(np.array(hand_pose["right_shape"],np.float32))
        print(random_pose.shape)
        print(random_shape.shape)
        print(random_tran.shape)

        mano_right_points = mano_layer_right(random_pose, random_shape)
        r_hand_verts_scaled = mano_right_points[0] / 1000.0 + random_tran
        r_hand_verts_scaled = r_hand_verts_scaled.detach().numpy()[0]

        # ones = np.ones((r_hand_verts_scaled.shape[0], 1))
        # r_hand_verts_homogeneous = np.hstack((r_hand_verts_scaled, ones))  # Shape: (N, 4)

        # # Remove the fourth column to make it Nx3
        # r_hand_verts_homogeneous = r_hand_verts_homogeneous[:, :3]

        # # Transpose to get 3xN for matrix multiplication
        # r_hand_verts_homogeneous = r_hand_verts_homogeneous.T  # Shape: (3, N)

        # # Project the points
        # r_image_points_homogeneous = intrinsics @ r_hand_verts_homogeneous  # Shape: (3, N)

        # # Normalize by the third row to get pixel coordinates
        # r_image_points = r_image_points_homogeneous[:2, :] / r_image_points_homogeneous[2, :]

        # # Transpose to get Nx2 array
        # r_points_2d = r_image_points.T 

        # l_faces = mano_layer_left.th_faces.numpy()
        # r_faces = mano_layer_right.th_faces.numpy()

        intensity = 0
        img_size = test_im.shape
        obj_points = mesh_points[k]
        obj_verts = mesh_verts[k]
        obj_faces = mesh_faces
        # l_points_dist= []
        # for i in range(l_points_2d.shape[0]):
        #     p = l_hand_verts_scaled[i]
        #     dist = compute_dist_to_mesh(p, obj_points)
        #     #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
        #     l_points_dist.append(dist)
        # r_points_dist= []
        # for i in range(r_hand_verts_scaled.shape[0]):
        #     p = r_hand_verts_scaled[i]
        #     dist = compute_dist_to_mesh(p, obj_points)
        #     #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
        #     r_points_dist.append(dist)
        

        # l_image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        # r_image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        # l_depth_buffer = np.full((img_size[0], img_size[1]), 0)
        # r_depth_buffer = np.full((img_size[0], img_size[1]), 0)

        # for face in l_faces:
        #     pts = []
        #     dists = []
        #     for elem in face:
        #         pts.append(l_points_2d[elem])
        #         dists.append(l_points_dist[elem])
        #     pts = np.array(pts)
        #     pts = np.array(pts, dtype=np.int32)  # Ensure the points are of type int32
        #     pts = pts.reshape((-1, 1, 2)) 
        #     mean_dist = np.mean(dists)
        #     face_depth = get_intensity(mean_dist)*255
        #     # Create a mask for the face
        #     mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        #     cv2.fillPoly(mask, [pts], color=1)
        #     mask_indices = np.where(mask == 1)
        #     for y, x in zip(*mask_indices):
        #         if face_depth > l_depth_buffer[y, x]:
        #             l_depth_buffer[y, x] = face_depth
        #             l_image[y, x] = (face_depth, face_depth, face_depth)

        # for face in r_faces:
        #     pts = []
        #     dists = []
        #     for elem in face:
        #         pts.append(r_points_2d[elem])
        #         dists.append(r_points_dist[elem])
        #     pts = np.array(pts)
        #     pts = np.array(pts, dtype=np.int32)  # Ensure the points are of type int32
        #     pts = pts.reshape((-1, 1, 2)) 
        #     mean_dist = np.mean(dists)
        #     face_depth = get_intensity(mean_dist)*255
        #     # Create a mask for the face
        #     mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
        #     cv2.fillPoly(mask, [pts], color=1)
        #     mask_indices = np.where(mask == 1)
        #     for y, x in zip(*mask_indices):
        #         if face_depth > r_depth_buffer[y, x]:
        #             r_depth_buffer[y, x] = face_depth
        #             r_image[y, x] = (face_depth, face_depth, face_depth)
        
        # r_save = r_image[:,:,0]
        # l_save = l_image[:,:,0]

        # combined_save = np.maximum(r_save, l_save).astype(np.uint8)
        # hand_masks.append(combined_save)

        


        

        # combined_image = cv2.addWeighted(l_image, 1.0, r_image, 1.0, 0)
             

        ones = np.ones((obj_verts.shape[0],1))
        obj_verts_homogeneous = np.hstack((obj_verts, ones))  # Shape: (N, 4)
        # Remove the fourth column to make it Nx3
        obj_verts_homogeneous = obj_verts_homogeneous[:, :3]

        # Transpose to get 3xN for matrix multiplication
        obj_verts_homogeneous = obj_verts_homogeneous.T  # Shape: (3, N)

        # Project the points
        obj_image_points_homogeneous = intrinsics @ obj_verts_homogeneous  # Shape: (3, N)

        # Normalize by the third row to get pixel coordinates
        obj_image_points = obj_image_points_homogeneous[:2, :] / obj_image_points_homogeneous[2, :]

        # Transpose to get Nx2 array
        obj_verts_2d = obj_image_points.T 
        
        both_hand_verts = np.vstack((l_hand_verts_scaled, r_hand_verts_scaled))
        obj_verts_dist= []
        for i in range(obj_verts.shape[0]):
            p = obj_verts[i]
            dist = compute_dist_to_mesh(p, both_hand_verts)
            #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
            obj_verts_dist.append(dist)




        obj_image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        obj_depth_buffer = np.full((img_size[0], img_size[1]), 0)
    
        for face in obj_faces:
            pts = []
            dists = []
            for elem in face:
                pts.append(obj_verts_2d[elem])
                dists.append(obj_verts_dist[elem])
            pts = np.array(pts)
            pts = np.array(pts, dtype=np.int32)  # Ensure the points are of type int32
            pts = pts.reshape((-1, 1, 2)) 
            mean_dist = np.mean(dists)
            face_depth = get_intensity(mean_dist)*255
            # Create a mask for the face
            mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], color=1)
            mask_indices = np.where(mask == 1)
            for y, x in zip(*mask_indices):
                if face_depth > obj_depth_buffer[y, x]:
                    obj_depth_buffer[y, x] = face_depth
                    obj_image[y, x] = (face_depth, face_depth, face_depth)



        obj_mask = np.array(obj_image[:,:,0]).astype(np.uint8)
        obj_masks.append(obj_mask)
        # comb_mask = cv2.addWeighted(combined_image, 1.0, obj_mask, 1.0, 0)

        output_image = cv2.addWeighted(obj_image, 0.7,test_im, 1.0, 0)

        
        
       
        

        gif.append(output_image)
    

    
    # np.save(save_dir_hands + f'{idx:03d}.npy', np.array(hand_masks))
    # np.save(save_dir_obj + f'{idx:03d}.npy', np.array(obj_masks))
    print(f'Finished {idx:03d}')
    imageio.mimsave(gif_path +f'{idx:03d}.gif', gif, duration = 1000)

        # Display the result
    