import numpy as np
import cv2
import png
from open3d import geometry, utility, pipelines
from utils.registration import icp
import os
from utils.tools import get_camera_parameters_local
import glob

def load_images(path, ID):
    camera_intrinsics, _ = get_camera_parameters_local()
    LABEL_INTERVAL = 1
    
    img_file = path + 'images/%s.jpg' % (ID*LABEL_INTERVAL)
    img_rgb = cv2.imread(img_file)

    mask_file = path + 'masks/%s.png' % (ID*LABEL_INTERVAL)
    img_mask = cv2.imread(mask_file, 0)
    # aplica a máscara na imagem usando a função bitwise_and
    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask)

    depth_file = path + 'depth/%s.png' % (ID*LABEL_INTERVAL)
    img_depth = png.Reader(depth_file)
    pngdata = img_depth.read()
    depth = np.array(tuple(map(np.uint16, pngdata[2])))
    pointcloud = convert_depth_frame_to_pointcloud(depth, camera_intrinsics, img_mask)

    return (img_rgb, pointcloud, masked_img)


def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics, mask):
    [height, width] = depth_image.shape

    # Aplica a máscara na imagem de profundidade
    depth_image_masked = np.where(mask == 0, 0, depth_image)

    # Código restante é o mesmo
    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() -
         float(camera_intrinsics['ppx']))/float(camera_intrinsics['fx'])
    y = (v.flatten() -
         float(camera_intrinsics['ppy']))/float(camera_intrinsics['fy'])
    depth_image_masked = depth_image_masked*float(camera_intrinsics['depth_scale'])
    z = depth_image_masked.flatten()
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    pointcloud = np.dstack((x, y, z)).reshape(
        (depth_image.shape[0], depth_image.shape[1], 3))

    return pointcloud

def marker_registration(source, target):
     img_rgb_src, depth_src = source
     img_rgb_des, depth_des = target
 
     gray_src = cv2.cvtColor(img_rgb_src, cv2.COLOR_RGB2GRAY)
     gray_des = cv2.cvtColor(img_rgb_des, cv2.COLOR_RGB2GRAY)

     aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250) # Aqui era 250
     parameters = cv2.aruco.DetectorParameters()
     detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
     #lists of ids and the corners beloning to each id
     corners_src, _ids_src, _ = detector.detectMarkers(gray_src)
     corners_des, _ids_des, _ = detector.detectMarkers(gray_des)

     try:
         ids_src = []
         ids_des = []
         for i in range(len(_ids_src)):
              ids_src.append(_ids_src[i][0])
         for i in range(len(_ids_des)):
              ids_des.append(_ids_des[i][0])
     except:
         return None

     common = [x for x in ids_src if x in ids_des]
  
     if len(common) < 2:
          # too few marker matches, use icp instead
          return None
     
     src_good = []
     dst_good = []
     for i, id in enumerate(ids_des):
          if id in ids_src:
               j = ids_src.index(id)
               for count, corner in enumerate(corners_src[j][0]):
                    feature_3D_src = depth_src[int(corner[1])][int(corner[0])]
                    feature_3D_des = depth_des[int(corners_des[i][0][count][1])][int(corners_des[i][0][count][0])]
                    if feature_3D_src[2]!=0 and feature_3D_des[2]!=0:
                         src_good.append(feature_3D_src)
                         dst_good.append(feature_3D_des)
    
     # get rigid transforms between 2 set of feature points through ransac
     try:
          transform, _ = cv2.estimateAffine3D(np.asarray(src_good), np.asarray(dst_good), cv2.FM_RANSAC)
          return transform
     except:
          return None

def load_pointcloud(path, Filename, downsample = True, interval = 1):
     voxel_size = 0.001 
     camera_intrinsics, _ = get_camera_parameters_local() 
    
     img_file = path + 'images/%s.jpg' % (Filename*interval)

     img_rgb = cv2.imread(img_file)
     img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
     mask_file = path + 'masks/%s.png' % (Filename*interval)
     img_mask = cv2.imread(mask_file, 0)

     #masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask)

     depth_file = path + 'depth/%s.png' % (Filename*interval)
     img_depth = png.Reader(depth_file)
     pngdata = img_depth.read()
     depth = np.array(tuple(map(np.uint16, pngdata[2])))
     mask = depth.copy()
     depth = convert_depth_frame_to_pointcloud(depth, camera_intrinsics, img_mask)

     source = geometry.PointCloud()
     source.points = utility.Vector3dVector(depth[mask>0])
     source.colors = utility.Vector3dVector(img_rgb[mask>0])

     if downsample == True:
          source = source.voxel_down_sample(voxel_size = voxel_size)
          source.estimate_normals(geometry.KDTreeSearchParamHybrid(radius = 0.002 * 2, max_nn = 30))
       
     return source

def registration(path,max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):

     N_Neighbours = 10
     voxel_size = 0.001
     ICP_METHOD = 'point-to-plane'

     n_pcds = len(os.listdir(path + 'images/'))

     pose_graph = pipelines.registration.PoseGraph()
     odometry = np.identity(4)
     pose_graph.nodes.append(pipelines.registration.PoseGraphNode(odometry))
     
     pcds = [[] for i in range(n_pcds)]
     
     for source_id in range(n_pcds):
          if source_id > 0:
               pcds[source_id-1] = []
          step = max(1,int(n_pcds/N_Neighbours))
          for target_id in range(source_id + 1, n_pcds, step):
               
               # derive pairwise registration through feature matching
               color_src, depth_src, mask_src  = load_images(path, source_id)
               color_dst, depth_dst, mask_dst  = load_images(path, target_id)
               res = marker_registration((color_src, depth_src),
                                      (color_dst, depth_dst))

               if res is not None:
                    print(f'res is not None in source_id {source_id} and target_id {target_id}')
     
               if res is None and target_id != source_id + 1:
                    # ignore such connections
                    continue

               if not pcds[source_id]:
                    pcds[source_id] = load_pointcloud(path, source_id, downsample = True)
               if not pcds[target_id]:
                    pcds[target_id] = load_pointcloud(path, target_id, downsample = True)
               if res is None:
                    # if marker_registration fails, perform pointcloud matching
                    transformation_icp, information_icp = icp(
                         pcds[source_id], pcds[target_id], voxel_size, max_correspondence_distance_coarse,
                         max_correspondence_distance_fine, method = ICP_METHOD)

               else:
                    transformation_icp = res
                    information_icp = pipelines.registration.get_information_matrix_from_point_clouds(
                         pcds[source_id], pcds[target_id], max_correspondence_distance_fine,
                         transformation_icp)

               if target_id == source_id + 1:
                    # odometry
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    pose_graph.edges.append(pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                          transformation_icp, information_icp, uncertain = False))
               else:
                    # loop closure
                    pose_graph.edges.append(pipelines.registration.PoseGraphEdge(source_id, target_id,
                                                          transformation_icp, information_icp, uncertain = True))

     return pose_graph

if __name__ == "__main__":
    voxel_size = 0.001
    LABEL_INTERVAL = 1
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    
    path = 'urna/'

    camera_intrinsics = get_camera_parameters_local()

    Ts = []

    n_pcds = int(len(glob.glob1(path+"images","*.jpg"))/LABEL_INTERVAL)
    print("Full registration ...")
    pose_graph = registration(path, max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option =pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance = max_correspondence_distance_fine,
            edge_prune_threshold = 0.25,
            reference_node = 0)
    pipelines.registration.global_optimization(pose_graph,
                                        pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                                        pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)


    num_annotations = int(len(glob.glob1(path+"images","*.jpg"))/LABEL_INTERVAL)

    for point_id in range(num_annotations):
            Ts.append(pose_graph.nodes[point_id].pose)
    Ts = np.array(Ts)
    filename = path + 'transforms.npy'
    np.save(filename, Ts)
    print("Transforms saved")