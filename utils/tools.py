from json import load
import numpy as np


def get_annotations(path):
    try:
        annotations = load(open(path, 'r'))
        # list of dictionaries
        annotations = annotations['annotations']
        values = {}

        for annotation in annotations:
            image_id = str(annotation['image_id'])
            bbox = annotation['bbox']
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            values[image_id] = bbox

        return values
    except Exception as e:
        print("Problem to get annotations: ", e)
        return None

def get_camera_parameters(path_intrinsic_json='utils/intrinsics.json'):
    camera_matrix = None
    distortion_coefficients = None
    try:
        with open(path_intrinsic_json, 'r') as fp:
            camera_parameters = load(fp)

        camera_matrix = np.array([[camera_parameters['fx'], 0, camera_parameters['ppx']], [0, camera_parameters['fy'], camera_parameters['ppy']], [0, 0, 1]])
        distortion_coefficients = np.array(camera_parameters['distortion'])
    except Exception as e:
        print("Problem to get camera parameters: ", e)

    return camera_matrix, distortion_coefficients