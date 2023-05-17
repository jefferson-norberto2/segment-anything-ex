import cv2
import numpy as np
#import open3d as o3d
#from utils.tools import get_camera_parameters
import trimesh 
from utils.tools import get_camera_parameters_local
import os
import random

def compute_projection(points_3D,internal_calibration):
    points_3D = points_3D.T
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

folder = "urna/"
path_label = folder + "label/"

if not os.path.exists(path_label):
    os.makedirs(path_label)

LABEL_INTERVAL = 1
camera_intrinsics, K = get_camera_parameters_local()

transforms_file = folder + 'transforms.npy'
path_transforms = folder + "transforms"
if not os.path.exists(path_transforms):
    os.makedirs(path_transforms)

try:
    transforms = np.load(transforms_file)
except:
    print("transforms not computed, run compute_gt_poses.py first")
    exit()

mesh = trimesh.load(folder + "urna.ply")
Tform = mesh.apply_obb() # need sicpy
mesh.export(file_obj = folder + folder[8:-1] +".ply")

points = mesh.bounding_box.vertices
center = mesh.centroid
min_x = np.min(points[:,0])
min_y = np.min(points[:,1])
min_z = np.min(points[:,2])
max_x = np.max(points[:,0])
max_y = np.max(points[:,1])
max_z = np.max(points[:,2])
points = np.array([[min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z],
                    [min_x, max_y, max_z], [max_x, min_y, min_z], [max_x, min_y, max_z],
                    [max_x, max_y, min_z], [max_x, max_y, max_z]])

points_original = np.concatenate((np.array([[center[0],center[1],center[2]]]), points))
points_original = trimesh.transformations.transform_points(points_original,
                                                            np.linalg.inv(Tform))

projections = [[],[]]


for i in range(len(transforms)):
    mesh_copy = mesh.copy()
    img = cv2.imread(folder+"images/" + str(i*LABEL_INTERVAL) + ".jpg")
    transform = np.linalg.inv(transforms[i])
    transformed = trimesh.transformations.transform_points(points_original, transform)

    
    corners = compute_projection(transformed,K)
    corners = corners.T
    corners[:,0] = corners[:,0]/int(camera_intrinsics['width'])
    corners[:,1] = corners[:,1]/int(camera_intrinsics['height'])

    T = np.dot(transform, np.linalg.inv(Tform))
    mesh_copy.apply_transform(T)
    filename = path_transforms + "/"+ str(i*LABEL_INTERVAL)+".npy"
    np.save(filename, T)
    
    #sample_points = mesh_copy.sample(10000)
    dellist = [j for j in range(0, len(mesh_copy.vertices))]
    dellist = random.sample(dellist, len(mesh_copy.vertices) - 10000)
    sample_points = np.delete(mesh_copy.vertices, dellist, axis=0)
    
    masks = compute_projection(sample_points,K)
    masks = masks.T

    min_x = np.min(masks[:,0])
    min_y = np.min(masks[:,1])
    max_x = np.max(masks[:,0])
    max_y = np.max(masks[:,1])
    
    file = open(path_label+"/"+ str(i*LABEL_INTERVAL)+".txt","w")
    
    message = str(0)[:8] + " "
    file.write(message)
    for pixel in corners:
        for digit in pixel:
            message = str(digit)[:8]  + " "
            file.write(message)
    message = str((max_x-min_x)/float(camera_intrinsics['width']))[:8]  + " "
    file.write(message) 
    message = str((max_y-min_y)/float(camera_intrinsics['height']))[:8]
    file.write(message)
    file.close()
