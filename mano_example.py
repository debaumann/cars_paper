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
fx = 636.6593017578125/2 
fy = 636.251953125/2
u_0 = 635.283881879317/2
v_0 = 366.8740353496978/2
intrinsics = np.array([[fx, 0, u_0], [0, fy, v_0], [0, 0, 1]])

batch_size = 1
# Select number of principal components for pose space
ncomps = 48

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
idxs = [0,33,66,99,121]
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
    test_mesh = np.asarray(get_mesh_vertices(idx))
    


    left = test_data[:,:62]
    right = test_data[:,62:]

    l_translation = left[:,1:4]
    l_pose = left[:,4:4+48]
    print(np.shape(l_pose))
    
    l_shape = left[:,4+48:]

    r_translation = right[:,1:4]
    r_pose = right[:,4:4+48]
    r_shape = right[:,4+48:]

    ############### if visualize points in 3d plot ###############
    # l_test_translation = torch.tensor(l_translation[1]).unsqueeze(0).float()
    # l_test_shape = torch.tensor(l_shape[1]).unsqueeze(0).float()
    # l_test_pose = torch.tensor(l_pose[1]).unsqueeze(0).float()

    # gt_points = mano_layer_left(l_test_pose, l_test_shape)
    # pred_mano = torch.tensor(pred_pose[1]).unsqueeze(0).float()
    # pred_points = mano_layer_left(pred_mano, l_test_shape)
    
    # # Convert points to numpy arrays for plotting
    # gt_points_np = gt_points[0].squeeze(0).detach().numpy()
    # pred_points_np = pred_points[0].squeeze(0).detach().numpy()
    # gt_joints_np = gt_points[1].squeeze(0).detach().numpy()
    # pred_joints_np = pred_points[1].squeeze(0).detach().numpy()
    # plot_two_hands(gt_points_np, pred_points_np, gt_joints_np, pred_joints_np)
    ###############################################################


    gif = []
    pose_losses = []
    points_losses = []
    joints_losses = []

    ############start of sequence ################
    for k in range(8):

        curr_cam_pose = test_pose[k].reshape(4,4)
        ######## need this for loss untainted ########
        l_gt_pose = torch.tensor(l_pose[k]).unsqueeze(0).float()
        pred_mano_loss_pose = torch.tensor(pred_pose[k]).unsqueeze(0).float()
    
        
        
        #########rotate and translate mano parameters ########

        left_rot = R.from_rotvec(l_pose[k][:3])
        l_translation_col_vec = np.array(l_translation[k]).reshape(3, 1)
        left_mat = np.concatenate((np.concatenate((left_rot.as_matrix(), l_translation_col_vec), axis=1), [[0, 0, 0, 1]]), axis=0)
        left_mat_proj = np.dot(curr_cam_pose, left_mat)
        left_rotvec = R.from_matrix(left_mat_proj[:3, :3]).as_rotvec()
        l_pose[k][:3] = left_rotvec
        

        right_rot = R.from_rotvec(r_pose[k][:3])
        r_translation_col_vec = np.array(r_translation[k]).reshape(3, 1)
        right_mat = np.concatenate((np.concatenate((right_rot.as_matrix(), r_translation_col_vec), axis=1), [[0, 0, 0, 1]]), axis=0)
        right_mat_proj = np.dot(curr_cam_pose, right_mat)
        right_rotvec = R.from_matrix(right_mat_proj[:3, :3]).as_rotvec()
        r_pose[k][:3] = right_rotvec

        pred_rot = R.from_rotvec(pred_pose[k][:3])
        pred_translation_col_vec = np.array(l_translation[k]).reshape(3, 1)
        pred_mat = np.concatenate((np.concatenate((pred_rot.as_matrix(), pred_translation_col_vec), axis=1), [[0, 0, 0, 1]]), axis=0)
        pred_mat_proj = np.dot(curr_cam_pose, pred_mat)
        pred_rotvec = R.from_matrix(pred_mat_proj[:3, :3]).as_rotvec()
        pred_pose[k][:3] = pred_rotvec

        ######## load new mano parameters ########
    
        l_mano_pose = torch.tensor(l_pose[k]).unsqueeze(0).float()
        l_mano_shape = torch.tensor(l_shape[k]).unsqueeze(0).float()
        l_mano_translation = torch.tensor(l_translation[k]).unsqueeze(0).float()

        r_mano_pose = torch.tensor(r_pose[k]).unsqueeze(0).float()
        r_mano_shape = torch.tensor(r_shape[k]).unsqueeze(0).float()
        r_mano_translation = torch.tensor(r_translation[k]).unsqueeze(0).float()

        pred_mano_pose = torch.tensor(pred_pose[k]).unsqueeze(0).float()
        pred_mano_shape = torch.tensor(l_shape[k]).unsqueeze(0).float()
        pred_mano_translation = torch.tensor(l_translation[k]).unsqueeze(0).float()

        #########forward pass through mano layer and new trans and scale ########
        ########left hand ########
        mano_keypoints_3d_left = mano_layer_left(l_mano_pose, l_mano_shape)
        left_hand_origin = mano_keypoints_3d_left[0][0][0] / 1000.0
        origin_left = torch.unsqueeze(
            left_hand_origin, 1) + l_mano_translation.T
        left_mat_proj = np.dot(
            curr_cam_pose, np.concatenate((origin_left, [[1]])))
        new_left_trans = torch.tensor(
            left_mat_proj.T[0, :3]) - left_hand_origin
        
        l_verts = mano_keypoints_3d_left[0]/1000 + new_left_trans
        l_joints = mano_keypoints_3d_left[1]/1000 + new_left_trans

        ########right hand ########
        mano_keypoints_3d_right = mano_layer_right(r_mano_pose, r_mano_shape)
        right_hand_origin = mano_keypoints_3d_right[0][0][0] / 1000.0
        origin_right = torch.unsqueeze(
            right_hand_origin, 1) + r_mano_translation.T
        right_mat_proj = np.dot(
            curr_cam_pose, np.concatenate((origin_right, [[1]])))
        new_right_trans = torch.tensor(
            right_mat_proj.T[0, :3]) - right_hand_origin

        r_verts = mano_keypoints_3d_right[0]/1000 + new_right_trans
        r_joints = mano_keypoints_3d_right[1]/1000 + new_right_trans

        ########pred hand, curr only left ########
        mano_keypoints_3d_pred = mano_layer_left(pred_mano_pose, pred_mano_shape)
        pred_hand_origin = mano_keypoints_3d_pred[0][0][0] / 1000.0
        origin_pred = torch.unsqueeze(
            pred_hand_origin, 1) + pred_mano_translation.T
        pred_mat_proj = np.dot(
            curr_cam_pose, np.concatenate((origin_pred, [[1]])))
        new_pred_trans = torch.tensor(
            pred_mat_proj.T[0, :3]) - pred_hand_origin

        pred_verts = mano_keypoints_3d_pred[0]/1000 + new_pred_trans
        pred_joints = mano_keypoints_3d_pred[1]/1000 + new_pred_trans

        ########compute losses ########
        pose_loss = F.mse_loss(l_gt_pose, pred_mano_loss_pose)
        points_loss = F.mse_loss(mano_keypoints_3d_left[0], mano_keypoints_3d_pred[0])
        joints_loss = F.mse_loss(mano_keypoints_3d_left[1], mano_keypoints_3d_pred[1])
        print('pose loss',pose_loss.item(), 'verts loss', points_loss.item(), 'joints loss', joints_loss.item())
        #load image and project points into it
        points_losses.append(points_loss.item())
        pose_losses.append(pose_loss.item())
        joints_losses.append(joints_loss.item())


        ####### reprojection ########
        test_im = test_image[k].astype(np.uint8)
        obj_p = test_obj[k]
        obj_p = np.delete(obj_p, 0)
        obj_p = np.reshape(obj_p, (21,3))
        bbox_corners = np.array([obj_p[1], obj_p[2], obj_p[3], obj_p[4],obj_p[5], obj_p[6], obj_p[7], obj_p[8]])
        bbox_center = np.array(obj_p[0])

        ######mano output to numpy ########
        l_points = l_verts.squeeze(0).unsqueeze(1).detach().numpy()
        r_points = r_verts.squeeze(0).unsqueeze(1).detach().numpy()
        pred_points = pred_verts.squeeze(0).unsqueeze(1).detach().numpy()

        l_joints = l_joints.squeeze(0).unsqueeze(1).detach().numpy()
        r_joints = r_joints.squeeze(0).unsqueeze(1).detach().numpy()
        pred_joints = pred_joints.squeeze(0).unsqueeze(1).detach().numpy()

        ####### project points into image ########

        l_points_2d= []
        l_points_dist = []
        l_use_translation = l_mano_translation.squeeze(0).detach().numpy()
        l_use_translation = np.zeros_like(l_mano_translation.squeeze(0).detach().numpy())

        pred_points_2d= []
        pred_points_dist = []
        pred_use_translation = pred_mano_translation.squeeze(0).detach().numpy()
        pred_use_translation = np.zeros_like(pred_mano_translation.squeeze(0).detach().numpy())

        r_points_2d= []
        r_points_dist = []
        r_use_translation = r_mano_translation.squeeze(0).detach().numpy()
        r_use_translation = np.zeros_like(r_mano_translation.squeeze(0).detach().numpy())

        for i in range(l_points.shape[0]):
            p = l_points[i]
            pc, p3 = project_points(p,intrinsics,curr_cam_pose, l_use_translation) 
            dist = compute_dist_to_mesh(p3, test_mesh[k])
            #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
            p_c = pc[0]
            u, v = int(p_c[0]), int(p_c[1])
            l_points_2d.append([u,v])
            l_points_dist.append(dist)

        for i in range(pred_points.shape[0]):
            p = pred_points[i]
            pc, p3 = project_points(p,intrinsics,curr_cam_pose, pred_use_translation) 
            dist = compute_dist_to_mesh(p3, test_mesh[k])
            #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
            p_c = pc[0]
            u, v = int(p_c[0]), int(p_c[1])
            pred_points_2d.append([u,v])
            pred_points_dist.append(dist)
        
        for i in range(r_points.shape[0]):
            p = r_points[i]
            pc, p3 = project_points(p,intrinsics,curr_cam_pose, r_use_translation)
            dist = compute_dist_to_mesh(p3, test_mesh[k])
            #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
            p_c = pc[0]
            u, v = int(p_c[0]), int(p_c[1])
            r_points_2d.append([u,v])
            r_points_dist.append(dist)


        l_faces = mano_layer_left.th_faces.numpy()

        intensity = 0
        img_size = test_im.shape

        l_image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        l_depth_buffer = np.full((img_size[0], img_size[1]), 0)

        for face in l_faces:
            pts = []
            dists = []
            for elem in face:
                pts.append(l_points_2d[elem])
                dists.append(l_points_dist[elem])
            pts = np.array(pts)
            mean_dist = np.mean(dists)
            face_depth = get_intensity(mean_dist)*255
            # Create a mask for the face
            mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], color=1)
            
            # Update the image and depth buffer
            mask_indices = np.where(mask == 1)
            for y, x in zip(*mask_indices):
                if face_depth > l_depth_buffer[y, x]:
                    l_depth_buffer[y, x] = face_depth
                    l_image[y, x] = (0, face_depth, 0)
        
        pred_faces = mano_layer_left.th_faces.numpy()

        pred_image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        pred_depth_buffer = np.full((img_size[0], img_size[1]), 0)

        for face in pred_faces:
            pts = []
            dists = []
            for elem in face:
                pts.append(pred_points_2d[elem])
                dists.append(pred_points_dist[elem])
            pts = np.array(pts)
            mean_dist = np.mean(dists)
            face_depth = get_intensity(mean_dist)*255
            # Create a mask for the face
            mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], color=1)
            
            # Update the image and depth buffer
            mask_indices = np.where(mask == 1)
            for y, x in zip(*mask_indices):
                if face_depth > pred_depth_buffer[y, x]:
                    pred_depth_buffer[y, x] = face_depth
                    pred_image[y, x] = (0, 0, 255)

        r_faces = mano_layer_right.th_faces.numpy()

        intensity = 0
        img_size = test_im.shape

        r_image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        r_depth_buffer = np.full((img_size[0], img_size[1]), 0)

        for face in r_faces:
            pts = []
            dists = []
            for elem in face:
                pts.append(r_points_2d[elem])
                dists.append(r_points_dist[elem])
            pts = np.array(pts)
            mean_dist = np.mean(dists)
            face_depth = get_intensity(mean_dist)*255.0
            # Create a mask for the face
            mask = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], color=1)
            
            # Update the image and depth buffer
            mask_indices = np.where(mask == 1)
            for y, x in zip(*mask_indices):
                if face_depth > r_depth_buffer[y, x]:
                    r_depth_buffer[y, x] = face_depth
                    r_image[y, x] = (0, face_depth, 0)

        # Combine the left and right images
        combined_image = np.maximum(l_image, r_image)         
        combined_image = cv2.addWeighted(combined_image, 1.0, pred_image , 1.0, 0)
        output_image = cv2.addWeighted(combined_image, 1.0,test_im, 1.0, 0)

        ##########compute and draw joints ########
        l_joints_2d= []
        r_joints_2d= []
        pred_joints_2d= []
        for i in range(l_joints.shape[0]):
            p = l_joints[i]
            pc, p3 = project_joint_points(p,intrinsics,curr_cam_pose, l_use_translation) 
            p_c = pc[0]
            u, v = int(p_c[0]), int(p_c[1])
            l_joints_2d.append([u,v])

            p = r_joints[i]
            pc, p3 = project_joint_points(p,intrinsics,curr_cam_pose, r_use_translation)
            p_c = pc[0]
            u, v = int(p_c[0]), int(p_c[1])
            r_joints_2d.append([u,v])


            p = pred_joints[i]
            pc, p3 = project_joint_points(p,intrinsics,curr_cam_pose, pred_use_translation)
            p_c = pc[0]
            u, v = int(p_c[0]), int(p_c[1])
            pred_joints_2d.append([u,v])

        for i in range(len(l_joints_2d)):
            u, v = l_joints_2d[i]
            cv2.circle(output_image, (u, v), 2, (0, 0, 255), -1)
            u, v = pred_joints_2d[i]
            cv2.circle(output_image, (u, v), 2, (255, 0, 0), -1)
            u, v = r_joints_2d[i]
            cv2.circle(output_image, (u, v), 2, (0, 255, 0), -1)
        

        gif.append(output_image)
    pose_losses = np.array(pose_losses)
    points_losses = np.array(points_losses)
    joints_losses = np.array(joints_losses)

    # Compute mean and standard deviation for each loss
    pose_mean, pose_std = np.mean(pose_losses), np.std(pose_losses)
    points_mean, points_std = np.mean(points_losses), np.std(points_losses)
    joints_mean, joints_std = np.mean(joints_losses), np.std(joints_losses)

    # Normalize the losses
    normalized_pose_losses = (pose_losses - pose_mean) / pose_std
    normalized_points_losses = (points_losses - points_mean) / points_std
    normalized_joints_losses = (joints_losses - joints_mean) / joints_std
    print('Normalized pose loss:', normalized_pose_losses)
    print('Normalized verts loss:', normalized_points_losses)
    print('Normalized joints loss:', normalized_joints_losses)
    imageio.mimsave(gif_path +f'{idx:03d}.gif', gif, duration = 1000)

        # Display the result
    