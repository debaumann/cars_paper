import os
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import tqdm
from typing import List

fx = 636.6593017578125/2 
fy = 636.251953125/2
u_0 = 635.283881879317/2
v_0 = 366.8740353496978/2
intrinsics = np.array([[fx, 0, u_0], [0, fy, v_0], [0, 0, 1]])


obj_mesh_path = '/Users/dennisbaumann/cars_paper/data/object/'
obj_pose_path = '/Users/dennisbaumann/cars_paper/data/train/obj_rt_8_val/'
action_labels = np.load('/Users/dennisbaumann/cars_paper/data/action_labels_train.npy')
print(action_labels[0])

action_to_object = {
    0: 'background',
    1: 'book',
    2: 'espresso',
    3: 'lotion',
    4: 'spray',
    5: 'milk',
    6: 'cocoa',
    7: 'chips',
    8: 'cappuccino',
    9: 'book',
    10: 'espresso',
    11: 'lotion',
    12: 'spray',
    13: 'milk',
    14: 'cocoa',
    15: 'chips',
    16: 'cappuccino',
    17: 'lotion',
    18: 'milk',
    19: 'chips',
    20: 'lotion',
    21: 'milk',
    22: 'chips',
    23: 'milk',
    24: 'espresso',
    25: 'cocoa',
    26: 'chips',
    27: 'cappuccino',
    28: 'espresso',
    29: 'cocoa',
    30: 'cappuccino',
    31: 'lotion',
    32: 'spray',
    33: 'book',
    34: 'espresso',
    35: 'spray',
    36: 'lotion'
}


def apply_transformation_to_points(points, transformation):
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = np.dot(transformation, points_hom.T).T
    return transformed_points[:, :3]

def get_object_mesh(obj_mesh_path: str, obj_pose_path: str) -> o3d.geometry.TriangleMesh:
        obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
        return obj_mesh


def simplify_mesh(mesh, target_number_of_triangles):
    simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles)
    return simplified_mesh

def get_mesh_vertices(idx: int) -> np.ndarray:
    str_id = f"{idx:03d}"
    print(str_id)
    action = action_labels[idx]
    obj_id = action_to_object[action]
    print(obj_id)
    obj_mesh_path = f'/Users/dennisbaumann/cars_paper/data/object/{obj_id}/{obj_id}.obj'
    transformation = np.load(f'/Users/dennisbaumann/cars_paper/data/train/obj_rt_8_train/{str_id}.npy')


    
    transformation = np.delete(transformation, 0, axis=1)



    
    
    obj_mesh = get_object_mesh(obj_mesh_path, obj_pose_path)
    obj_vertices = np.asarray(obj_mesh.vertices)
    faces = np.asarray(obj_mesh.triangles)
    print('n of faces',np.shape(faces))
    print('n of vertices',np.shape(obj_vertices))
    obj_sample = np.asarray(obj_mesh.sample_points_uniformly(number_of_points=1000).points)
    print(np.shape(obj_sample))
    stacked_transformation = []
    simp_transformations = []
    for t in transformation:
        transfo = t.reshape(4, 4)
        new_mesh = apply_transformation_to_points(obj_sample, transfo)
        new_simp = apply_transformation_to_points(obj_vertices, transfo)

        simp_transformations.append(new_simp)
        stacked_transformation.append(new_mesh)
    return np.array(stacked_transformation), np.array(simp_transformations), np.array(faces)

