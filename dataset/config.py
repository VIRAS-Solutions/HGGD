import logging

import numpy as np
import open3d as o3d

camera = 'realsense'
logging.warning(f'Using {camera} camera now!')


def set_current_camera_intrinsics(intrinsics):
    global current_camera_intrinsics
    current_camera_intrinsics = intrinsics

def get_camera_intrinsic(camera=camera):
    # Set dynammically camera instrinsics from ImageRGBD ROS Topic
    if current_camera_intrinsics is not None:
        return current_camera_intrinsics
    
    logging.warning("Using fallback camera intrinsics - no dynamic values available!")
    #Fallback to hardcoded instrinsics
    if camera == 'kinect':
        intrinsics = np.array([[631.55, 0, 638.43], [0, 631.21, 366.50],
                               [0, 0, 1]])
    # elif camera == 'realsense':
    #     intrinsics = np.array([[927.17, 0, 651.32], [0, 927.37, 349.62],
    #                            [0, 0, 1]])
    elif camera == 'realsense':
        intrinsics = np.array([[609.7703857421875, 0, 325.3979187011719], 
                              [0, 609.0634155273438, 243.4080810546875],
                              [0, 0, 1]])
    else:
        raise ValueError(
            'Camera format must be either "kinect" or "realsense".')
    return intrinsics


# # realsense
# intrinsics = np.array([[927.16973877, 0, 651.31506348],
#                        [0, 927.36688232, 349.62133789], [0, 0, 1]])

# # kinect
# intrinsics = np.array([[631.55, 0, 638.43],
#                        [0, 631.21, 366.50], [0, 0, 1]])
