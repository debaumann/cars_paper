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


obj_mesh_path = '/Users/dennisbaumann/cars_paper/data/arctic_data/meta/object_vtemplates/box'
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

def get_object_mesh(obj_mesh_path: str) -> o3d.geometry.TriangleMesh:
        obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
        return obj_mesh


def simplify_mesh(mesh, target_number_of_triangles):
    simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles)
    return simplified_mesh

def get_mesh_vertices() -> np.ndarray:
    
    obj_mesh_path = f'/Users/dennisbaumann/cars_paper/data/arctic_data/meta/object_vtemplates/box/'
    obj_top = obj_mesh_path + 'top.obj'
    obj_bottom = obj_mesh_path + 'bottom.obj'


    



    
    
    obj_mesh_top= get_object_mesh(obj_top)
    obj_vertices_top = np.asarray(obj_mesh_top.vertices)
    faces_top = np.asarray(obj_mesh_top.triangles)
    
    print('n of vertices',np.shape(obj_vertices_top))
    obj_sample_top = np.asarray(obj_mesh_top.sample_points_uniformly(number_of_points=1000).points)
    print(np.shape(obj_sample_top))

    obj_mesh_bottom= get_object_mesh(obj_bottom)
    obj_vertices_bottom = np.asarray(obj_mesh_bottom.vertices)
    faces_bottom = np.asarray(obj_mesh_bottom.triangles)
    
    print('n of vertices',np.shape(obj_vertices_bottom))
    obj_sample_bottom = np.asarray(obj_mesh_bottom.sample_points_uniformly(number_of_points=1000).points)

    
    return np.array(faces_top), np.array(faces_bottom), np.array(obj_vertices_top), np.array(obj_vertices_bottom)

