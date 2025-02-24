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


obj_mesh_path = '/cluster/home/debaumann/cars_paper/arctic_data/meta/object_vtemplates/box'
#obj_pose_path = '/Users/dennisbaumann/cars_paper/data/train/obj_rt_8_val/'
#action_labels = np.load('/Users/dennisbaumann/cars_paper/data/action_labels_train.npy')
#print(action_labels[0])

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


# def get_mesh_vertices():
#     """
#     Loads a simplified mesh from an OBJ file and uses a parts.json file to split the faces.
    
#     The parts.json file is assumed to be a list (or array) where each element corresponds to a vertex,
#     with a value of 1 for 'top' and 0 for 'bottom'.
    
#     Returns:
#         top_faces: (N_top, k) array of face vertex indices for the top part
#         bottom_faces: (N_bottom, k) array of face vertex indices for the bottom part
#         verts: (M, 3) array of vertex coordinates
#         parts: (M,) array of part labels (1 for top, 0 for bottom)
#     """
#     mesh_file_path = '/Users/dennisbaumann/cars_paper/data/arctic_data/meta/object_vtemplates/box/mesh.obj'
#     parts_file_path= '/Users/dennisbaumann/cars_paper/data/arctic_data/meta/object_vtemplates/box/parts.json'

 
#     # Load vertices and faces from the OBJ file.
#     verts = []
#     faces = []
#     with open(mesh_file_path, 'r') as f:
#         for line in f:
#             if line.startswith("v "):
#                 # Parse vertex line and convert coordinates to float.
#                 vals = line.strip().split()[1:]
#                 verts.append([float(x) for x in vals])
#             elif line.startswith("f "):
#                 # Parse face line.
#                 # OBJ indices are 1-indexed so subtract 1.
#                 face = [int(token.split('/')[0]) - 1 for token in line.strip().split()[1:]]
#                 faces.append(face)
#     verts = np.array(verts, dtype=np.float64)
#     if verts.ndim == 1:
#         verts = verts.reshape(-1, 3)
#     faces = np.array(faces)
    
#     # Load parts labels from the JSON file.
#     with open(parts_file_path, 'r') as f:
#         parts_labels = np.array(json.load(f))
#     parts_labels = parts_labels.flatten()  # Ensure it's a 1D array.
    
#     # Create masks and get indices for top and bottom vertices.
#     top_mask = (parts_labels == 1)
#     bottom_mask = (parts_labels == 0)
#     top_indices = np.where(top_mask)[0]
#     bottom_indices = np.where(bottom_mask)[0]
    
#     # Extract vertices for each part.
#     top_verts = verts[top_indices, :]   # shape: (N_top, 3)
#     bottom_verts = verts[bottom_indices, :]  # shape: (N_bottom, 3)
    
#     # Create mapping dictionaries from original indices to new indices.
#     top_index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(top_indices)}
#     bottom_index_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(bottom_indices)}
    
#     # Separate faces into top and bottom.
#     top_faces = []
#     bottom_faces = []
#     for face in faces:
#         # Check if all vertices of the face are top or all bottom.
#         if np.all(top_mask[face]):
#             # Remap indices for top face.
#             remapped_face = [top_index_map[v] for v in face]
#             top_faces.append(remapped_face)
#         elif np.all(bottom_mask[face]):
#             # Remap indices for bottom face.
#             remapped_face = [bottom_index_map[v] for v in face]
#             bottom_faces.append(remapped_face)
#         # If a face is mixed (contains both top and bottom vertices), you can choose to:
#         # - Discard it, or
#         # - Assign it by majority vote (with additional remapping logic),
#         # Here we discard mixed faces.
    
#     top_faces = np.array(top_faces)
#     bottom_faces = np.array(bottom_faces)
    
#     return top_faces,  bottom_faces,top_verts,bottom_verts
def extract_obj_name(adress): 
    parts = adress.split('/')
    
    obj_name = parts[1].split('_')[0]
    return obj_name

def get_mesh_vertices(adress, target_number_of_triangles=2000) -> np.ndarray:
    obj_name = extract_obj_name(adress)
    obj_mesh_path = f'/cluster/home/debaumann/cars_paper/arctic_data/meta/object_vtemplates/{obj_name}/'
    obj_top = obj_mesh_path + 'top.obj'
    obj_bottom = obj_mesh_path + 'bottom.obj'

    # Load and simplify the top mesh
    obj_mesh_top = get_object_mesh(obj_top)
    obj_mesh_top = simplify_mesh(obj_mesh_top, target_number_of_triangles)
    obj_vertices_top = np.asarray(obj_mesh_top.vertices)
    faces_top = np.asarray(obj_mesh_top.triangles)
    
    print('Number of vertices (top):', np.shape(obj_vertices_top))
    print('Number of faces (top):', np.shape(faces_top))

    # Load and simplify the bottom mesh
    obj_mesh_bottom = get_object_mesh(obj_bottom)
    obj_mesh_bottom = simplify_mesh(obj_mesh_bottom, target_number_of_triangles)
    obj_vertices_bottom = np.asarray(obj_mesh_bottom.vertices)
    faces_bottom = np.asarray(obj_mesh_bottom.triangles)
    
    print('Number of vertices (bottom):', np.shape(obj_vertices_bottom))
    print('Number of faces (bottom):', np.shape(faces_bottom))

    return np.array(faces_top), np.array(faces_bottom), np.array(obj_vertices_top), np.array(obj_vertices_bottom)
