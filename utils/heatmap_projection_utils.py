import numpy as np
from numpy.typing import NDArray  
from typing import Tuple
import matplotlib.pyplot as plt
import open3d as o3d







def plot_two_hands(gt_hand: np.ndarray, pred_hand: np.ndarray, ax: plt.Axes) -> None:
    """ Plot two hands in 3D. """
    gt_points_np = gt_hand
    pred_points_np = pred_hand
    print(gt_points_np.shape)
    print(pred_points_np.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Plot ground truth points
    ax.scatter(gt_points_np[:, 0], gt_points_np[:, 1], gt_points_np[:, 2], c='b', marker='o', label='Ground Truth')
    #ax.scatter(gt_joints_np[:, 0], gt_joints_np[:, 1], gt_joints_np[:, 2], c='g',s = 30, marker='o', label='Ground Truth Joints')
    # Plot predicted points
    ax.scatter(pred_points_np[:, 0], pred_points_np[:, 1], pred_points_np[:, 2], c='y', marker='^', label='Predicted')
    #ax.scatter(pred_joints_np[:, 0], pred_joints_np[:, 1], pred_joints_np[:, 2], s=30, c='r', marker='^', label='Predicted Joints')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Ground Truth vs Predicted Hand Points')
    plt.show()
    # # demo.display_hand({'verts': mano_points[0], 'joints': mano_points[1]}, mano_faces=mano_layer_left.th_faces)
    
    
def project_points(points_3d: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    world_to_camera = np.linalg.inv(cam_to_world)
    points_cam = np.dot(world_to_camera, points_3d_hom.T).T 
    points_cam[:, :3] += translation
    projected = np.dot(intrinsics, points_cam[:, :3].T)
    projected[:2] /= projected[2]
    return projected[:2].T, points_cam[:, :3]

def project_joint_points(points_3d: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    world_to_camera = cam_to_world
    points_cam = np.dot(world_to_camera, points_3d_hom.T).T 
    points_cam[:, :3] += translation
    projected = np.dot(intrinsics, points_cam[:, :3].T)
    projected[:2] /= projected[2]
    return projected[:2].T, points_cam[:, :3]

def line_plane_intersection(line_point: np.ndarray, 
                            line_dir: np.ndarray, 
                            plane_point: np.ndarray, 
                            plane_normal: np.ndarray) -> np.ndarray:
    """ Calculate the intersection of a line with a plane. """
    denom = np.dot(plane_normal, line_dir)
    if np.abs(denom) < 1e-6:
        # Line and plane are parallel
        return None
    d = np.dot(plane_point - line_point, plane_normal) / denom
    intersection = line_point + d * line_dir
    return intersection


def is_point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    """ Check if a point is inside a polygon defined by vertices on a plane. """
    total_angle = 0.0
    for i in range(len(vertices)):
        a, b = vertices[i], vertices[(i + 1) % len(vertices)]
        da, db = a - point, b - point
        angle = np.arctan2(np.linalg.norm(np.cross(da, db)), np.dot(da, db))
        total_angle += angle
    return np.isclose(total_angle, 2 * np.pi, atol=1e-5)

def dist_to_bbox(bbox_corners: np.ndarray, 
                 bbox_center: np.ndarray, 
                 hand_point: np.ndarray) -> Tuple[float, np.ndarray]:
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


def get_intensity(dist: float) -> float:
    weight = 1 / dist
    if weight > 50:
        weight = 50
    return weight / 50

def compute_dist_to_mesh(point: np.ndarray, mesh: np.ndarray) -> float:
    distances = np.linalg.norm(mesh - point, axis=1)
    dist = np.min(distances)
    return dist