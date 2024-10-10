import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np
import matplotlib.pyplot as plt
import sys 
import os 
import cv2


#camera params 
fx = 636.6593017578125/2 
fy = 636.251953125/2
u_0 = 635.283881879317/2
v_0 = 366.8740353496978/2
intrinsics = np.array([[fx, 0, u_0], [0, fy, v_0], [0, 0, 1]])

batch_size = 1
# Select number of principal components for pose space
ncomps = 48

# Initialize MANO layer
mano_layer_left = ManoLayer(mano_root='/Users/dennisbaumann/manopth/mano/models', use_pca=False, flat_hand_mean=True, ncomps=ncomps, side='left')
mano_layer_right= ManoLayer(mano_root='/Users/dennisbaumann/manopth/mano/models', use_pca=False,flat_hand_mean=True, ncomps=ncomps, side='right')
manos_data_path = '/Users/dennisbaumann/cars_paper/data/mano_8_train/'
cam_data_path = '/Users/dennisbaumann/cars_paper/data/cam_8_train/'
obj_data_path = '/Users/dennisbaumann/cars_paper/data/obj_8_train/'
# Generate random shape parameters
test_data = np.load(os.path.join(manos_data_path, '001.npy'))
test_pose = np.load(os.path.join(cam_data_path, '001.npy'))
test_obj = np.load(os.path.join(obj_data_path, '001.npy'))


left = test_data[:,:62]
right = test_data[:,62:]

l_translation = left[:,1:4]
l_pose = left[:,4:4+48]
l_shape = left[:,4+48:]

r_translation = right[:,1:4]
r_pose = right[:,4:4+48]
r_shape = right[:,4+48:]

for k in range(8):
    l_test_translation = torch.tensor(l_translation[k]).unsqueeze(0).float()
    l_test_shape = torch.tensor(l_shape[k]).unsqueeze(0).float()
    l_test_pose = torch.tensor(l_pose[k]).unsqueeze(0).float()

    r_test_translation = torch.tensor(r_translation[k]).unsqueeze(0).float()
    r_test_shape = torch.tensor(r_shape[k]).unsqueeze(0).float()
    r_test_pose = torch.tensor(r_pose[k]).unsqueeze(0).float()

    # Forward pass through MANO layer
    l_output = mano_layer_left(l_test_pose, l_test_shape)
    l_verts = l_output[0]/1000
    l_joints = l_output[1]/1000
    l_verts += l_test_translation
    l_joints += l_test_translation

    r_output = mano_layer_right(r_test_pose, r_test_shape)
    r_verts = r_output[0]/1000
    r_joints = r_output[1]/1000

    #demo.display_hand({'verts': r_verts, 'joints': r_joints}, mano_faces=mano_layer_right.th_faces)
    r_verts += r_test_translation
    r_joints += r_test_translation

    def project_points(points_3d, intrinsics, cam_to_world):
        # Convert points to homogeneous coordinates
        points_3d_hom = np.hstack((np.array(points_3d), np.ones((1, 1))))
        
        # Transform points from world to camera coordinates
        points_cam = np.dot(cam_to_world, points_3d_hom.T).T
        points_in_cam = points_cam[:, :3]
        
        # Project points using intrinsics
        projected = np.dot(intrinsics, points_cam[:, :3].T)
        
        # Normalize by the third row
        projected[:2] /= projected[2]
        
        return projected[:2].T , points_in_cam # Return only x, y coordinates



    def line_plane_intersection(line_point, line_dir, plane_point, plane_normal):
        """ Calculate the intersection of a line with a plane. """
        denom = np.dot(plane_normal, line_dir)
        if np.abs(denom) < 1e-6:
            # Line and plane are parallel
            return None
        d = np.dot(plane_point - line_point, plane_normal) / denom
        intersection = line_point + d * line_dir
        return intersection

    def is_point_in_polygon(point, vertices):
        """ Check if a point is inside a polygon defined by vertices on a plane. """
        total_angle = 0
        for i in range(len(vertices)):
            a, b = vertices[i], vertices[(i + 1) % len(vertices)]
            da, db = a - point, b - point
            angle = np.arctan2(np.linalg.norm(np.cross(da, db)), np.dot(da, db))
            total_angle += angle
        return np.isclose(total_angle, 2 * np.pi, atol=1e-5)


    def dist_to_bbox(bbox_corner,bbox_center, hand_point):
        line_dir = bbox_center - hand_point
        line_dir /= np.linalg.norm(line_dir)  # Normalize the direction vector
        
        # Define bounding box faces using corner indices
        faces = [
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), 
            (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)
        ]
        min_distance = np.inf
        closest_point = None
        
        for indices in faces:
            vertices = bbox_corners[list(indices)]
            plane_normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[1])
            plane_normal /= np.linalg.norm(plane_normal)  # Normalize
            intersection = line_plane_intersection(hand_point, line_dir, vertices[0], plane_normal)
            if intersection is not None and is_point_in_polygon(intersection, vertices):
                distance = np.linalg.norm(intersection - hand_point)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = intersection
        
        return min_distance, closest_point

    def get_intensity(dist):
        weight = 1/dist
        if weight>50:
            weight = 50
        return weight/50


    #load image and project points into it
    test_im = np.load('/Users/dennisbaumann/cars_paper/data/seq_8_train/frames_train(1)/001.npy')
    test_im = test_im[k]

    obj_p = test_obj[k]
    print(obj_p.shape)
    obj_p = np.delete(obj_p, 0)
    obj_p = np.reshape(obj_p, (21,3))
    bbox_corners = np.array([obj_p[1], obj_p[2], obj_p[3], obj_p[4],obj_p[5], obj_p[6], obj_p[7], obj_p[8]])
    bbox_center = np.array(obj_p[0])

    print(test_im.shape)
    l_points = l_verts.squeeze(0).unsqueeze(1).detach().numpy()
    r_points = r_verts.squeeze(0).unsqueeze(1).detach().numpy()
    curr_cam_pose = test_pose[k].reshape(4,4)
    print(l_points.shape)
    l_points_2d= []
    l_points_dist = []
    for i in range(l_points.shape[0]):
        p = l_points[i]
        pc, p3 = project_points(p,intrinsics,curr_cam_pose)
        dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
        p_c = pc[0]
        u, v = int(p_c[0]), int(p_c[1])
        l_points_2d.append([u,v])
        l_points_dist.append(dist)

    r_points_2d= []
    r_points_dist = []
    for i in range(r_points.shape[0]):
        p = r_points[i]
        pc, p3 = project_points(p,intrinsics,curr_cam_pose)
        dist , closest_point = dist_to_bbox(bbox_corners,bbox_center, p3[0])
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
        face_depth = get_intensity(mean_dist)*255
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

    # Add the combined image to the test image
    output_image = cv2.addWeighted(test_im, 1.0, combined_image, 1.0, 0)

    # Display the result
    plt.imshow(output_image)
    plt.axis('off')
    plt.imsave(f'/Users/dennisbaumann/cars_paper/mano_test_output/mesh_overlay_{k}.png', output_image)
    plt.clf()