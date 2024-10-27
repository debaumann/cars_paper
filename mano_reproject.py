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

gif_path = '/Users/dennisbaumann/cars_paper/mano_test_output_val/'
os.makedirs(gif_path, exist_ok=True)
# Initialize MANO layer
mano_layer_left = ManoLayer(mano_root='/Users/dennisbaumann/manopth/mano/models', use_pca=False, flat_hand_mean=True, ncomps=ncomps, side='left')
mano_layer_right= ManoLayer(mano_root='/Users/dennisbaumann/manopth/mano/models', use_pca=False,flat_hand_mean=True, ncomps=ncomps, side='right')
manos_data_path = '/Users/dennisbaumann/cars_paper/data/val/mano_8_val/'

mano_pred_path = '/Users/dennisbaumann/cars_paper/data/mano_val/'
cam_data_path = '/Users/dennisbaumann/cars_paper/data/val/cam_8_val/'
obj_data_path = '/Users/dennisbaumann/cars_paper/data/val/obj_8_val/'
#joint_data_path = '/Users/dennisbaumann/cars_paper/data/val/joint_8_val/'
obj_rt_data_path = '/Users/dennisbaumann/cars_paper/data/val/obj_rt_8_val/'
idxs = np.arange(0, 121)
# Generate random shape parameters
for idx in idxs:
    test_data = np.load(os.path.join(manos_data_path, f'{idx:03d}.npy'))
    test_pose = np.load(os.path.join(cam_data_path, f'{idx:03d}.npy'))
    test_obj = np.load(os.path.join(obj_data_path, f'{idx:03d}.npy'))
    #test_joints = np.load(os.path.join(joint_data_path, f'{idx:03d}.npy'))
    test_obj_rt = np.load(os.path.join(obj_rt_data_path, f'{idx:03d}.npy'))
    test_obj_rt = np.delete(test_obj_rt, 0, axis=1)

    pred_pose = np.load(os.path.join(mano_pred_path, f'{idx:03d}.npy'))
    print(np.shape(pred_pose))
    test_image = np.load(f'/Users/dennisbaumann/cars_paper/data/val/seq_8_val/{idx:03d}.npy')
    mesh_points, mesh_verts, mesh_faces= get_mesh_vertices(idx)
    
    
    gif = []
    pose_losses = []
    points_losses = []
    joints_losses = []

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

        ones = np.ones((l_hand_verts_scaled.shape[0], 1))
        l_hand_verts_homogeneous = np.hstack((l_hand_verts_scaled, ones))  # Shape: (N, 4)

        # Remove the fourth column to make it Nx3
        l_hand_verts_homogeneous = l_hand_verts_homogeneous[:, :3]

        # Transpose to get 3xN for matrix multiplication
        l_hand_verts_homogeneous = l_hand_verts_homogeneous.T  # Shape: (3, N)

        # Project the points
        l_image_points_homogeneous = intrinsics @ l_hand_verts_homogeneous  # Shape: (3, N)

        # Normalize by the third row to get pixel coordinates
        l_image_points = l_image_points_homogeneous[:2, :] / l_image_points_homogeneous[2, :]

        # Transpose to get Nx2 array
        l_points_2d = l_image_points.T 



        random_pose = torch.tensor(np.array(hand_pose["right_pose"],np.float32))
        random_tran = torch.tensor(np.array(hand_pose["right_tran"],np.float32))
        random_shape = torch.tensor(np.array(hand_pose["right_shape"],np.float32))


        mano_right_points = mano_layer_right(random_pose, random_shape)
        r_hand_verts_scaled = mano_right_points[0] / 1000.0 + random_tran
        r_hand_verts_scaled = r_hand_verts_scaled.detach().numpy()[0]

        ones = np.ones((r_hand_verts_scaled.shape[0], 1))
        r_hand_verts_homogeneous = np.hstack((r_hand_verts_scaled, ones))  # Shape: (N, 4)

        # Remove the fourth column to make it Nx3
        r_hand_verts_homogeneous = r_hand_verts_homogeneous[:, :3]

        # Transpose to get 3xN for matrix multiplication
        r_hand_verts_homogeneous = r_hand_verts_homogeneous.T  # Shape: (3, N)

        # Project the points
        r_image_points_homogeneous = intrinsics @ r_hand_verts_homogeneous  # Shape: (3, N)

        # Normalize by the third row to get pixel coordinates
        r_image_points = r_image_points_homogeneous[:2, :] / r_image_points_homogeneous[2, :]

        # Transpose to get Nx2 array
        r_points_2d = r_image_points.T 



        # left_rot = R.from_rotvec(load_hand[4:7])
        # right_rot = R.from_rotvec(load_hand[66:69])

        # left_mat = np.concatenate((np.concatenate((left_rot.as_matrix(), np.array(
        #     hand_pose['left_tran']).T), axis=1), [[0, 0, 0, 1]]), axis=0)

        # right_mat = np.concatenate((np.concatenate((right_rot.as_matrix(), np.array(
        #     hand_pose['right_tran']).T), axis=1), [[0, 0, 0, 1]]), axis=0)


        # extrinsic_matrix = test_pose[k].reshape(4, 4)
        # print('extrinsic dtype:', extrinsic_matrix.dtype)

        # left_mat_proj = np.dot(extrinsic_matrix, left_mat)
        # right_mat_proj = np.dot(extrinsic_matrix, right_mat)


        # left_rotvec = R.from_matrix(left_mat_proj[:3, :3]).as_rotvec()
        # hand_pose["left_pose"][0][:3] = left_rotvec

        # right_rotvec = R.from_matrix(right_mat_proj[:3, :3]).as_rotvec()
        # hand_pose["right_pose"][0][:3] = right_rotvec

        # mano_keypoints_3d_left = mano_layer_left(torch.tensor(np.array(hand_pose["left_pose"], np.float32)),
        #              torch.tensor(np.array(hand_pose["left_shape"], np.float32)))

        # left_hand_origin = mano_keypoints_3d_left[0][0][0] / 1000.0
        # origin_left = torch.unsqueeze(left_hand_origin, 1) + torch.tensor(np.array(hand_pose["left_tran"])).T

        # left_mat_proj = np.dot(extrinsic_matrix, np.concatenate((origin_left, [[1]])))
        # new_left_trans = torch.tensor(left_mat_proj.T[0, :3]) - left_hand_origin

        # mano_left_points = mano_layer_left(torch.tensor(np.array(hand_pose["left_pose"], np.float32)),
        #            torch.tensor(np.array(hand_pose["left_shape"], np.float32)))
        # hand_verts_scaled = mano_left_points[0] / 1000.0 + torch.tensor(np.array(hand_pose["left_tran"], np.float32))

        # hand_verts_scaled = hand_verts_scaled.detach().numpy()[0]
        # ones = np.ones((hand_verts_scaled.shape[0], 1))
        # hand_verts_homogeneous = np.hstack((hand_verts_scaled, ones))  # Shape: (N, 4)

        # # Remove the fourth column to make it Nx3
        # hand_verts_homogeneous = hand_verts_homogeneous[:, :3]

        # # Transpose to get 3xN for matrix multiplication
        # hand_verts_homogeneous = hand_verts_homogeneous.T  # Shape: (3, N)

        # # Project the points
        # image_points_homogeneous = intrinsics @ hand_verts_homogeneous  # Shape: (3, N)

        # # Normalize by the third row to get pixel coordinates
        # image_points = image_points_homogeneous[:2, :] / image_points_homogeneous[2, :]

        # # Transpose to get Nx2 array
        # #l_points_2d = image_points.T  # Shape: (N, 2)

        # mano_keypoints_3d_right = mano_layer_right(torch.tensor(np.array(hand_pose["right_pose"], np.float32)),
        #              torch.tensor(np.array(hand_pose["right_shape"], np.float32)))

        # right_hand_origin = mano_keypoints_3d_right[1][0][0] / 1000.0
        # origin_right = torch.unsqueeze(right_hand_origin, 1) + torch.tensor(np.array(hand_pose["right_tran"])).T

        # right_mat_proj = np.dot(extrinsic_matrix, np.concatenate((origin_right, [[1]])))
        # new_right_trans = torch.tensor(right_mat_proj.T[0, :3]) - right_hand_origin
        # hand_pose["right_tran"] = new_right_trans


        # random_pose = torch.tensor(np.array(hand_pose["right_pose"],np.float32))
        # random_tran = torch.tensor(np.array(hand_pose["right_tran"],np.float32))
        # random_shape = torch.tensor(np.array(hand_pose["right_shape"],np.float32))


        # mano_right_points = mano_layer_right(random_pose, random_shape)
        # hand_verts_scaled = mano_right_points[0] / 1000.0 + random_tran #torch.tensor(np.array(hand_pose["right_tran"], np.float32))
        # world_cam_extrinsic = np.linalg.inv(extrinsic_matrix)
        # hand_verts_scaled = hand_verts_scaled.detach().numpy()[0]
        # hand_verts_scaled = np.dot(world_cam_extrinsic[:3, :3], hand_verts_scaled.T).T + world_cam_extrinsic[:3, 3]

        # ones = np.ones((hand_verts_scaled.shape[0], 1))
        # hand_verts_homogeneous = np.hstack((hand_verts_scaled, ones))  # Shape: (N, 4)

        # # Remove the fourth column to make it Nx3
        # hand_verts_homogeneous = hand_verts_homogeneous[:, :3]

        # # Transpose to get 3xN for matrix multiplication
        # hand_verts_homogeneous = hand_verts_homogeneous.T  # Shape: (3, N)

        # # Project the points
        # image_points_homogeneous = intrinsics @ hand_verts_homogeneous  # Shape: (3, N)

        # # Normalize by the third row to get pixel coordinates
        # image_points = image_points_homogeneous[:2, :] / image_points_homogeneous[2, :]

        # # Transpose to get Nx2 array
        # r_points_2d = image_points.T  # Shape: (N, 2)







        
        ####### project points into image ########

    
        
        # for i in range(l_points.shape[0]):
        #     p = l_points[i]
        #     pc, p3 = project_points(p,intrinsics,curr_cam_pose, l_use_translation) 
        #     dist = compute_dist_to_mesh(p3, test_mesh[k])
        #     #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
        #     p_c = pc[0]
        #     u, v = int(p_c[0]), int(p_c[1])
        #     l_points_2d.append([u,v])
        #     l_points_dist.append(dist)

        


        l_faces = mano_layer_left.th_faces.numpy()
        r_faces = mano_layer_right.th_faces.numpy()

        intensity = 0
        img_size = test_im.shape
        obj_points = mesh_points[k]
        obj_verts = mesh_verts[k]
        obj_faces = mesh_faces
        l_points_dist= []
        for i in range(l_points_2d.shape[0]):
            p = l_hand_verts_scaled[i]
            dist = compute_dist_to_mesh(p, obj_points)
            #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
            l_points_dist.append(dist)
        r_points_dist= []
        for i in range(r_hand_verts_scaled.shape[0]):
            p = r_hand_verts_scaled[i]
            dist = compute_dist_to_mesh(p, obj_points)
            #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
            r_points_dist.append(dist)
        

        l_image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        r_image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        l_depth_buffer = np.full((img_size[0], img_size[1]), 0)
        r_depth_buffer = np.full((img_size[0], img_size[1]), 0)

        for face in l_faces:
            pts = []
            dists = []
            for elem in face:
                pts.append(l_points_2d[elem])
                dists.append(l_points_dist[elem])
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
                if face_depth > l_depth_buffer[y, x]:
                    l_depth_buffer[y, x] = face_depth
                    l_image[y, x] = (0, face_depth, 0)

        for face in r_faces:
            pts = []
            dists = []
            for elem in face:
                pts.append(r_points_2d[elem])
                dists.append(r_points_dist[elem])
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
                if face_depth > r_depth_buffer[y, x]:
                    r_depth_buffer[y, x] = face_depth
                    r_image[y, x] = (0, face_depth, 0)
        

        combined_image = cv2.addWeighted(l_image, 1.0, r_image, 1.0, 0)
       
        #combined_image = cv2.addWeighted(combined_image, 1.0, pred_image , 1.0, 0)
        
        

        # ones = np.ones((obj_points.shape[0],1))
        # print('obj_homogeneous:', obj_points.shape)
        # print('ones:', ones.shape)  
        # o_obj_homogeneous = np.hstack((obj_points, ones))  # Shape: (N, 4)
        # print('obj_homogeneous:', obj_homogeneous.shape)
        # # Remove the fourth column to make it Nx3
        # o_obj_homogeneous = o_obj_homogeneous[:, :3]

        # # Transpose to get 3xN for matrix multiplication
        # o_obj_homogeneous = o_obj_homogeneous.T  # Shape: (3, N)

        # # Project the points
        # o_image_points_homogeneous = intrinsics @ o_obj_homogeneous  # Shape: (3, N)

        # # Normalize by the third row to get pixel coordinates
        # o_image_points = o_image_points_homogeneous[:2, :] / o_image_points_homogeneous[2, :]

        # # Transpose to get Nx2 array
        # obj_points_2d = o_image_points.T 
        # # print('obj_points_2d:', obj_points_2d.shape)
        # # for i in range(obj_points_2d.shape[0]):
        # #     u,v = obj_points_2d[i]
        # #     u,v = int(u), int(v)
        # #     cv2.circle(outpu, (u, v), 2, (255, 255, 255), -1)

        ones = np.ones((obj_verts.shape[0],1))
        print('obj_verts_homogeneous:', obj_verts.shape)
        print('ones:', ones.shape)  
        obj_verts_homogeneous = np.hstack((obj_verts, ones))  # Shape: (N, 4)
        print('obj_verts_homogeneous:', obj_verts_homogeneous.shape)
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
        print('obj_verts_2d:', obj_verts_2d.shape)
        obj_mask = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        for face in obj_faces:

            pts = []
            for elem in face:
                pts.append(obj_verts_2d[elem])
            pts = np.array(pts, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(obj_mask, [pts], color=(255, 255, 0))

        comb_mask = cv2.addWeighted(combined_image, 1.0, obj_mask, 1.0, 0)

        output_image = cv2.addWeighted(comb_mask, 0.7,test_im, 1.0, 0)

        
        
        ##########compute and draw joints ########
        # l_joints_2d= []
        # r_joints_2d= []
        # pred_joints_2d= []
        # for i in range(l_joints.shape[0]):
        #     p = l_joints[i]
        #     pc, p3 = project_joint_points(p,intrinsics,curr_cam_pose, l_use_translation) 
        #     p_c = pc[0]
        #     u, v = int(p_c[0]), int(p_c[1])
        #     l_joints_2d.append([u,v])

        #     p = r_joints[i]
        #     pc, p3 = project_joint_points(p,intrinsics,curr_cam_pose, r_use_translation)
        #     p_c = pc[0]
        #     u, v = int(p_c[0]), int(p_c[1])
        #     r_joints_2d.append([u,v])


        #     p = pred_joints[i]
        #     pc, p3 = project_joint_points(p,intrinsics,curr_cam_pose, pred_use_translation)
        #     p_c = pc[0]
        #     u, v = int(p_c[0]), int(p_c[1])
        #     pred_joints_2d.append([u,v])

        # for i in range(len(l_joints_2d)):
        #     u, v = l_joints_2d[i]
        #     cv2.circle(output_image, (u, v), 2, (0, 0, 255), -1)
        #     u, v = pred_joints_2d[i]
        #     cv2.circle(output_image, (u, v), 2, (255, 0, 0), -1)
        #     u, v = r_joints_2d[i]
        #     cv2.circle(output_image, (u, v), 2, (0, 255, 0), -1)
        

        gif.append(output_image)
    
    imageio.mimsave(gif_path +f'{idx:03d}.gif', gif, duration = 1000)

        # Display the result
    