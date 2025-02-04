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
from utils.obj_mesh_arctic import get_mesh_vertices
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
mano_layer_left = ManoLayer(mano_root='/Users/dennisbaumann/manopth/mano/models', use_pca=False, flat_hand_mean=False, ncomps=ncomps, side='left')
mano_layer_right= ManoLayer(mano_root='/Users/dennisbaumann/manopth/mano/models', use_pca=False,flat_hand_mean=False, ncomps=ncomps, side='right')
manos_data_path = f'/Users/dennisbaumann/cars_paper/data/{mode}/mano_8_{mode}/'

path_to_data = '/Users/dennisbaumann/cars_paper/data/s01'
npy_files = sorted([f for f in os.listdir(path_to_data) if f.endswith('.npy')])
print(npy_files[0:5])
data = [np.load(os.path.join(path_to_data, f), allow_pickle=True) for f in npy_files[4:8]]

data_dicts = []
for obj in data:
    if isinstance(obj, np.ndarray) and obj.dtype == 'O':
        obj_dict = {key: value for key, value in obj.item().items()}
        data_dicts.append(obj_dict)
    else:
        data_dicts.append(obj)
mano_dict = data_dicts[1]
camera_dict = data_dicts[0]
print(mano_dict.keys())
image_dir = '/Users/dennisbaumann/cars_paper/data/arctic_data/box_use_01/0'
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

images = [cv2.imread(os.path.join(image_dir, f)) for f in image_files]
print(images[0].shape)

# Generate random shape parameters
save_dir_hands = f'/Users/dennisbaumann/cars_paper/data/arctic/{mode}/hand_masks/'
save_dir_obj = f'/Users/dennisbaumann/cars_paper/data/arctic/{mode}/obj_masks/'
os.makedirs(save_dir_hands, exist_ok=True)
os.makedirs(save_dir_obj, exist_ok=True)

length =732
mano_left= mano_dict['left']
mano_right = mano_dict['right']
distortions = np.array(camera_dict['dist8'])
R0= np.array(camera_dict['R0'])
T0 = np.array(camera_dict['T0'])*1000

object_arr = np.array(data_dicts[2])

mesh_faces_top, mesh_faces_bottom, mesh_verts_top,mesh_verts_bottom = get_mesh_vertices()
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
    print(pts_cam.shape,'shap pts cam')
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
    print(cam_pts_dist.shape,'shape cam pts dist')
    return cam_pts_dist

# Example usage


for idx in range(236,237):

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
    print(articulation_angle)
    rot_angle_axis = obj_array[1:4]
    obj_trans = obj_array[4:7]/1000
    obj_rot = R.from_rotvec(rot_angle_axis)
    obj_rot_matrix = obj_rot.as_matrix()
    obj_t = np.hstack((obj_rot_matrix,  (obj_trans).reshape(-1, 1)))
    canonical_T = np.hstack((R0, T0.reshape(-1, 1)))
    canonical_T = np.vstack((canonical_T, [0, 0, 0, 1]))
    print(f'obj_trans: {obj_trans}')
    obj_transformation = np.vstack((obj_t, [0, 0, 0, 1]))
    # obj_transformation = obj_transformation@canonical_T
    print(f'obj_transformation: {obj_transformation}')
    # Apply rotation and translation to the object mesh vertices
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
    mask = np.zeros((2000,2800, 3), dtype=np.uint8)
    l_mask = np.zeros((2000,2800, 3), dtype=np.uint8)
    r_mask = np.zeros((2000,2800, 3), dtype=np.uint8)
    l_faces = mano_layer_left.th_faces.numpy()
    r_faces = mano_layer_right.th_faces.numpy()

    l_verts = l_hand_verts_cam[:3,:].T
    r_verts = r_hand_verts_cam[:3,:].T
    print(l_verts.shape, r_verts.shape)
    obj_verts = np.hstack((obj_verts_top_cam[:3,:], obj_verts_bottom_cam[:3,:]))
    l_depth_buffer = np.full((2000, 2800), 0)
    r_depth_buffer = np.full((2000,2800), 0)
    l_points_dist= []
    for i in range(l_verts.shape[0]):
        p = l_verts[i]
        dist = compute_dist_to_mesh(p, obj_verts.T)
        #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
        l_points_dist.append(dist)
    r_points_dist= []
    for i in range(r_verts.shape[0]):
        p = r_verts[i]
        dist = compute_dist_to_mesh(p, obj_verts.T)
        #dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
        r_points_dist.append(dist)
    l_verts_img = l_hand_verts_img.T
    r_verts_img = r_hand_verts_img.T
    print(l_verts_img.shape, r_verts_img.shape)

    for face in l_faces:
            pts = []
            dists = []
            for elem in face:
                pts.append(l_verts_img[elem])
                dists.append(l_points_dist[elem])
            pts = np.array(pts)
            pts = np.array(pts, dtype=np.int32)  # Ensure the points are of type int32
            pts = pts.reshape((-1, 1, 2)) 
            mean_dist = np.mean(dists)
            face_depth = get_intensity(mean_dist)*255
            # Create a mask for the face
            mask = np.zeros((2000,2800), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], color=1)
            mask_indices = np.where(mask == 1)
            for y, x in zip(*mask_indices):
                if face_depth > l_depth_buffer[y, x]:
                    l_depth_buffer[y, x] = face_depth
                    l_mask[y, x] = (face_depth, face_depth, face_depth)
    for face in r_faces:
        pts = []
        dists = []
        for elem in face:
            pts.append(r_verts_img[elem])
            dists.append(r_points_dist[elem])
        pts = np.array(pts)
        pts = np.array(pts, dtype=np.int32)  # Ensure the points are of type int32
        pts = pts.reshape((-1, 1, 2)) 
        mean_dist = np.mean(dists)
        face_depth = get_intensity(mean_dist)*255
        # Create a mask for the face
        mask = np.zeros((2000,2800), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], color=1)
        mask_indices = np.where(mask == 1)
        for y, x in zip(*mask_indices):
            if face_depth > r_depth_buffer[y, x]:
                r_depth_buffer[y, x] = face_depth
                r_mask[y, x] = (face_depth, face_depth, face_depth)

    r_save = r_mask[:,:,0]
    l_save = l_mask[:,:,0]

    combined_save = np.maximum(r_save, l_save).astype(np.uint8)
    plt.imshow(combined_save, cmap='hot')
    plt.show()

    

    resized_img = cv2.resize(combined_save, (0, 0), fx=0.3, fy=0.3)
    crop_h = 600
    crop_w = 840
    # img = img[i_h//2-crop_h//2:i_h//2+crop_h//2, i_w//2-crop_w//2:i_w//2+crop_w//2, :]
    print(f'Max image position: ({np.max(l_hand_verts_img[0, :])}, {np.max(l_hand_verts_img[1, :])})')
    print(f'Min image position: ({np.min(l_hand_verts_img[0, :])}, {np.min(l_hand_verts_img[1, :])})')
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    overlay_img = cv2.addWeighted(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB), 0.5, resized_img, 0.5, 0)
    plt.imshow(overlay_img)
    plt.show()

    



        
        
       
        

    #     gif.append(output_image)
    

    
    # # np.save(save_dir_hands + f'{idx:03d}.npy', np.array(hand_masks))
    # np.save(save_dir_obj + f'{idx:03d}.npy', np.array(obj_masks))
    # print(f'Finished {idx:03d}')
    # imageio.mimsave(gif_path +f'{idx:03d}.gif', gif, duration = 1000)

    #     # Display the result
    