import numpy as np
import json


def get_camera_parameters(camera_parameter_dir):
    with open(camera_parameter_dir) as file:
        params = json.load(file)
    B = params['extrinsic']['baseline']
    fx = params['intrinsic']['fx']
    cx = params['intrinsic']['u0']
    cy = params['intrinsic']['v0']

    return B, fx, cx, cy


def depth_inpainting(disparity, camera_dir, left_image, left_image_uint8, right_image):
    invalid_map, depth_in_meter = depth_in_meter_with_invalid(disparity, camera_dir)
    left_disparity = (disparity.astype(np.float64) - 1)/256
    H, W, _ = left_image.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    points = np.concatenate((Y.reshape(-1,1), X.reshape(-1,1)), axis=1)

    epsilon = 12 / 255.0
    outlier_mask = outliers_photoconsistency(left_image, right_image, disparity, epsilon)


def depth_in_meter_with_invalid(depth_map, camera_dir):
    invalid_map = (depth_map == 0)
    disparity_in_pixel = (depth_map.astype(np.float64)-1)/256
    zero_map = (disparity_in_pixel == 0)
    B, fx, _, _ = get_camera_parameters(camera_dir)
    depth_in_meter = np.zeros(disparity_in_pixel.shape)
    depth_in_meter[~invalid_map and ~zero_map] = B*fx/disparity_in_pixel[~invalid_map and ~zero_map]
    depth_in_meter[invalid_map or zero_map] = float('inf')

    return invalid_map, depth_in_meter

def outliers_photoconsistency(left_image, right_image, left_disparity, epsilon):
    H, W, _ = left_image.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    X_aligned = np.round(X-left_disparity)
    inside_border = (X_aligned >= 1) and (X_aligned <= W)
    X_aligned = np.minimum(np.maximum(X_aligned, 1), W)

    np.ravel_multi_index()

    pass