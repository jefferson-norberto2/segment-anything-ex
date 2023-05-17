import numpy as np
from open3d import pipelines

def icp(source,target,voxel_size,max_correspondence_distance_coarse,max_correspondence_distance_fine,
        method = "colored-icp"):

    """
    Perform pointcloud registration using iterative closest point.

    Parameters
    ----------
    source : An open3d.Pointcloud instance
      6D pontcloud of a source segment
    target : An open3d.Pointcloud instance
      6D pointcloud of a target segment
    method : string
      colored-icp, as in Park, Q.-Y. Zhou, and V. Koltun, Colored Point Cloud 
      Registration Revisited, ICCV, 2017 (slower)
      point-to-plane, a coarse to fine implementation of point-to-plane icp (faster)
    max_correspondence_distance_coarse : float
      The max correspondence distance used for the course ICP during the process
      of coarse to fine registration (if point-to-plane)
    max_correspondence_distance_fine : float
      The max correspondence distance used for the fine ICP during the process 
      of coarse to fine registration (if point-to-plane)

    Returns
    ----------
    transformation_icp: (4,4) float
      The homogeneous rigid transformation that transforms source to the target's
      frame
    information_icp:
      An information matrix returned by open3d.get_information_matrix_from_ \
      point_clouds function
    """


    assert method in ["point-to-plane","colored-icp"],"point-to-plane or colored-icp"
    if method == "point-to-plane":
        icp_coarse = pipelines.registration.registration_icp(source, target,
                                                   max_correspondence_distance_coarse, np.identity(4),
                                                   pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = pipelines.registration.registration_icp(source, target,
                max_correspondence_distance_fine, icp_coarse.transformation,
                pipelines.registration.TransformationEstimationPointToPlane())

        transformation_icp = icp_fine.transformation


    if method == "colored-icp":
        result_icp = pipelines.registration.registration_colored_icp(source,target,voxel_size, np.identity(4),
                                                           pipelines.registration.ICPConvergenceCriteria(relative_fitness = 1e-8,
                                                                                               relative_rmse = 1e-8, max_iteration = 50))

        transformation_icp = result_icp.transformation

        
    information_icp = pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        transformation_icp)
    
    return transformation_icp, information_icp