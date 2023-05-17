import cv2
import numpy as np
import open3d
from utils.tools import get_camera_parameters_local
import trimesh

# Carregar arquivo .ply como objeto open3d.geometry.PointCloud
ply_file = "urna/urn.ply"
pcd = open3d.io.read_point_cloud(ply_file)
# Converter o objeto PointCloud em matriz numpy
point_cloud = np.asarray(pcd.points)

mesh = trimesh.load(ply_file)
Tform = mesh.apply_obb() # need sicpy
mesh.export(file_obj = ply_file +".ply")

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

# Parâmetros intrínsecos da câmera
camera_params, dist_coeffs = get_camera_parameters_local()
dist_coeffs = np.array([camera_params['distortion']]).astype(np.float32)
camera_params = np.array([[camera_params['fx'], 0, camera_params['ppx']], [0, camera_params['fy'], camera_params['ppy']], [0, 0, 1]]).astype(np.float32)

# Carregar imagem e máscara
img = cv2.imread("urna/images/0.jpg")
mask = cv2.imread("urna/masks/0.png", cv2.IMREAD_GRAYSCALE)

# Calcular os pontos projetados no espaço da imagem
image_points, _ = cv2.projectPoints(point_cloud, np.zeros((3,)), np.zeros((3,)), camera_params, dist_coeffs)

# Encontrar correspondências entre os pontos projetados e a máscara do objeto
src_good = []
dst_good = []
for i, p in enumerate(image_points):
    x, y = int(p[0][0]), int(p[0][1])
    m = mask[y, x]
    if mask[y, x] > 0:
        src_good.append(image_points[i])
        dst_good.append(np.array([x, y]))

# Converter os pontos para o sistema de coordenadas da câmera
src_good = np.asarray(src_good).astype(np.float32)
src_good = cv2.convertPointsToHomogeneous(src_good)
src_good = src_good[:, 0, :]
dst_good = np.asarray(dst_good).astype(np.float32)
dst_good = cv2.convertPointsToHomogeneous(dst_good)
dst_good = dst_good[:, 0, :]

# Imprimir informações sobre os pontos
print(src_good.shape, dst_good.shape)
print(type(src_good), type(dst_good))
print(type(camera_params), type(dist_coeffs))


# Obter a transformação rígida entre os pontos correspondentes através do RANSAC
_, rvec, tvec, inliers = cv2.solvePnPRansac(src_good, dst_good, camera_params, dist_coeffs)

# Converter o vetor de rotação em matriz de rotação
rot_matrix, _ = cv2.Rodrigues(rvec)

# Matriz de transformação 4x4
transform = np.eye(4)
transform[:3, :3] = rot_matrix
transform[:3, 3] = tvec.flatten()

print("Transformação 4x4:\n", transform)
