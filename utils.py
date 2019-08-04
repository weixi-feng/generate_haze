import numpy as np
import json
from skimage.segmentation import slic
from skimage import color
import math
import cv2
import pdb
import matplotlib.pyplot as plt
from imguidedfilter import *
from skimage.color import rgb2gray

INF = float('inf')


def disparity_in_pixels(disparity):
    return (disparity.astype(np.float64)-1)/256


def normal_equation(A, b):
    """
    :param A: assume A full rank
    :return: solve Ax=b by (A^T)Ax = (A^T)b ==> x = ((A^T)A)^(-1)(A^T)b
    """
    x = np.linalg.solve(A.T.dot(A), A.T.dot(b))
    return x


def clip_to_unit_range(image):
    clipped_image = np.minimum(np.maximum(image, 1), 0)
    return clipped_image


def mutual_distance(p1, p2):
    """
    Calculate mutual distance between two sets of points p1 and p2
    :param p1: (N, d)
    :param p2: (M, d)
    :return: mutual distance (N, M)
    """
    mutual = p1.dot(p2.T)
    D = np.sum(p1*p1, axis=1, keepdims=True) - 2*mutual + np.sum(p2*p2, axis=1)
    return D


def ismember(A, B):
    """
    Python version of ismember equivalent to MATLAB version
    :param A: numpy matrix of shape (N1, N2)
    :param B: numpy matrix of shape
    :return: boolean mask of same shape as A
    """
    mask = [a in B for Aa in A for a in Aa]
    mask = np.asarray(mask).reshape(A.shape)
    return mask


def get_camera_parameters(camera_parameter_dir):
    with open(camera_parameter_dir) as file:
        params = json.load(file)
    B = params['extrinsic']['baseline']
    fx = params['intrinsic']['fx']
    cx = params['intrinsic']['u0']
    cy = params['intrinsic']['v0']

    return B, fx, cx, cy


def distance_in_meter(depth, camera_path):
    _, fx, cx, cy = get_camera_parameters(camera_path)
    H, W = depth.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    distance_map = (fx**2 + (X-cx)**2 + (Y-cy)**2)/(fx**2)
    distance_map = depth*np.sqrt(distance_map)
    return distance_map


def inpaint_depth_with_plane(pixel_mask, plane):
    points = np.asarray(np.nonzero(pixel_mask == 1))
    points_homo = np.concatenate((points, np.ones((1, points.shape[1]))),
                                           axis=0)
    points_homo = points_homo.T
    inpaint_depth = np.maximum(points_homo.dot(plane), 0)
    return inpaint_depth


def depth_inpainting(disparity, camera_dir, left_image, left_image_uint8, right_image):
    """
    :param disparity:
    :param camera_dir:
    :param left_image:
    :param left_image_uint8:
    :param right_image:
    :return: main function to get depth map in meters
    """

    # get depth map with invalid pixels
    invalid_map, depth_in_meter = depth_in_meter_with_invalid(disparity, camera_dir)
    left_disparity = disparity_in_pixels(disparity)

    H, W, _ = left_image.shape
    # X, Y = np.meshgrid(np.arange(W), np.arange(H))
    # points = np.concatenate((Y.reshape(-1,1), X.reshape(-1,1)), axis=1)

    # outlier mask based on photo consistency
    epsilon = 12 / 255.0
    outlier_mask = get_outliers(left_image, right_image, left_disparity, epsilon)

    unreliable_mask = np.logical_or(outlier_mask, invalid_map).astype(np.bool)
    # SLIC segmentation
    n_segments = 2048
    compact = 10
    seg_mask = slic(left_image_uint8.transpose(1,0,2), n_segments, compact)
    seg_mask = seg_mask.transpose(1,0).astype(np.float64)
    true_num_segments = len(np.unique(seg_mask))

    # segment classification and plane fitting using RANSAC
    min_count_known = 20
    min_frac_known = 0.6
    max_depth = 50
    alpha = 1e-2
    iter = 2000
    p = 1 - 1e-2
    # whether the segment is visible region
    visible = np.zeros(true_num_segments, dtype=bool)
    # store plane parameters
    planes = np.zeros((true_num_segments, 3))
    # lab color space average
    lab_averages = np.zeros((true_num_segments, 3))
    # store centroids of segments
    centroids = np.zeros((true_num_segments, 2))
    # denote whether each segment is in an infinte plane
    is_plane_inifinte = np.zeros(true_num_segments, dtype=bool)

    ## convert rgb to CIELAB color space
    left_image_lab = color.rgb2lab(left_image)
    for i in range(true_num_segments):
        current_mask = (seg_mask==i)

        known_pixels = np.logical_and(current_mask, ~unreliable_mask)
        unreliable_pixels = np.logical_and(current_mask, unreliable_mask)

        # number of pixels = (number of unreliable) + (number of known)
        num_pixels = np.sum(current_mask)
        num_unreliable = np.sum(unreliable_pixels)
        num_known = num_pixels - num_unreliable

        if num_known >= np.maximum(min_count_known, min_frac_known*num_pixels):
            visible[i] = True
            # if true, then this segment is adequately visible.
            # RANSAC for plane fitting
            known_pixels_finite = np.logical_and(known_pixels, depth_in_meter<INF)
            num_finite = np.sum(known_pixels_finite)
            num_inf = num_known - num_finite

            if num_inf > num_finite:
                is_plane_inifinte[i] = 1
                planes[i, :] = INF
            else:
                # run RANSAC on finite pixels and see if the results is
                # still larger than infinite pixel number
                depth_in_meter_finite = depth_in_meter[known_pixels_finite]
                points = np.nonzero(known_pixels_finite == 1)
                plane_finite, inliers = RANSAC_plane_depth(points, depth_in_meter_finite, alpha, iter, p)
                inliers_count = len(inliers)

                if num_inf > inliers_count:
                    is_plane_inifinte[i] = True
                    planes[i, :] = INF
                else:
                    planes[i, :] = plane_finite

            # inpaint unreliable depth values using the fitted plane.
            depth_in_meter[unreliable_pixels] = inpaint_depth_with_plane(unreliable_pixels, planes[i, :])

            # re-inpaint the initially known depth values
            depth_to_inpaints = inpaint_depth_with_plane(known_pixels, planes[i, :])
            indices_known_pixels = np.asarray(np.nonzero(known_pixels == 1))
            large_outlier_pixels = np.abs((depth_to_inpaints - depth_in_meter[known_pixels])) > max_depth
            temp = indices_known_pixels[:, large_outlier_pixels]
            depth_in_meter[temp[0], temp[1]] = depth_to_inpaints[large_outlier_pixels]

        segment_lab = left_image_lab[current_mask, :]
        lab_averages[i, :] = np.mean(segment_lab, axis=0)
        segment_xy = np.asarray(np.nonzero(current_mask==1))
        centroids[i, :] = np.mean(segment_xy, axis=1)

    visible_indices = np.nonzero(visible == 1)[0]
    num_visible = len(visible_indices)
    invisible_indices = np.nonzero(visible == 0)[0]
    num_invisible = len(invisible_indices)

    # Assignment of unreliable segments to visible ones with greedy matching.
    lam = compact
    S = np.sqrt((H*W)/float(true_num_segments))
    lab_averages_invisible = lab_averages[invisible_indices, :]
    lab_averages_visible = lab_averages[visible_indices, :]
    centroids_invisible = centroids[invisible_indices, :]
    centroids_visible = centroids[visible_indices, :]

    lab_average_dists = mutual_distance(np.concatenate((lab_averages_invisible, lab_averages_visible), axis=0),
                                        lab_averages_invisible)
    centroid_dists = mutual_distance(np.concatenate((centroids_invisible, centroids_visible), axis=0),
                                     centroids_invisible)
    E = lab_average_dists + ((lam/S)**2)*centroid_dists
    idx_list = np.arange(0, true_num_segments*num_invisible, true_num_segments+1)
    idx_list = np.unravel_index(idx_list, dims=(true_num_segments, num_invisible))
    E[idx_list] = INF

    idx = np.argmin(E[num_invisible:, :], axis=0)
    E_min_vis = E[num_invisible:, :][idx, np.arange(num_invisible)]  # shape(num_visible,)
    E_min_invis = INF * np.ones(num_invisible)
    E_min = E_min_vis
    unmatched = np.ones(num_invisible, dtype=bool)
    best_match_is_visible = np.ones(num_invisible, dtype=bool)
    idx_invisible = idx
    idx_final = np.zeros(num_visible, dtype=np.int64)
    is_matched_with_visible = np.ones(num_invisible, dtype=bool)

    # Main loop for matching invisible segments to segments that have been assigned with planes
    while True in unmatched:
        unmatched_idx = np.nonzero(unmatched==1)[0]
        j_tmp = int(np.argmin(E_min[unmatched]))
        j = unmatched_idx[j_tmp]

        segment_current_id = invisible_indices[j]
        segment_current = (seg_mask==segment_current_id)
        unreliable_pixels = np.logical_and(segment_current, unreliable_mask)
        known_pixels = np.logical_and(segment_current, ~unreliable_mask)

        if best_match_is_visible[j]:
            idx_final[j] = idx[j]
            planes[segment_current_id, :] = planes[visible_indices[idx_final[j]], :]
            is_plane_inifinte[segment_current_id] = is_plane_inifinte[visible_indices[idx_final[j]]]
        else:
            idx_final[j] = idx_invisible[j]
            planes[segment_current_id, :] = planes[invisible_indices[idx_final[j]], :]
            is_plane_inifinte[segment_current_id] = is_plane_inifinte[invisible_indices[idx_final[j]]]
            is_matched_with_visible[j] = False

        depth_in_meter[unreliable_pixels] = inpaint_depth_with_plane(unreliable_pixels, planes[segment_current_id, :])
        depth_to_inpaints = inpaint_depth_with_plane(known_pixels, planes[segment_current_id, :])
        indices_known_pixels = np.asarray(np.nonzero(known_pixels==1))
        large_outlier_pixels = np.abs(depth_to_inpaints-depth_in_meter[known_pixels]) > max_depth
        temp = indices_known_pixels[:, large_outlier_pixels]
        depth_in_meter[temp[0], temp[1]] = depth_to_inpaints[large_outlier_pixels]

        # Update loop variables
        idx_invisible[E_min_invis > E[j, :]] = j
        E_min_invis = np.minimum(E_min_invis, E[j, :])
        E_min = np.minimum(E_min, E_min_invis)
        best_match_is_visible = E_min_vis <= E_min
        unmatched[j] = False

    wrong_segment_ids = np.unique(seg_mask[depth_in_meter<=0])
    is_depth_invalid = ismember(seg_mask, wrong_segment_ids)

    return depth_in_meter, is_depth_invalid


def depth_in_meter_with_invalid(input_disparity, camera_dir):
    invalid_mask = (input_disparity == 0)
    disparity_in_pixel = disparity_in_pixels(input_disparity)
    zero_mask = (disparity_in_pixel == 0)
    B, fx, _, _ = get_camera_parameters(camera_dir)
    depth_in_meter = np.zeros(disparity_in_pixel.shape)
    depth_in_meter[np.logical_and(~invalid_mask, ~zero_mask)] = \
        B*fx/disparity_in_pixel[np.logical_and(~invalid_mask, ~zero_mask)]
    depth_in_meter[np.logical_or(invalid_mask, zero_mask)] = float('inf')

    return invalid_mask, depth_in_meter


def get_outliers(left_image, right_image, left_disparity, epsilon):
    """
    :param left_image:
    :param right_image:
    :param left_disparity:
    :param epsilon:
    :return: get outliers by considering photo consistency
    """
    H, W, _ = left_image.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    X = X + 1
    X_aligned = np.round(X-left_disparity)
    inside_border = np.logical_and((X_aligned >= 1), (X_aligned <= W))
    X_aligned = (np.minimum(np.maximum(X_aligned, 1), W)-1).astype(Y.dtype)
    idx1, idx2 = Y.reshape(-1,), X_aligned.reshape(-1,)
    right_image_warped = right_image[idx1, idx2, :].reshape(H, W, 3)
    mask = np.logical_not(np.logical_and(
        np.sum((left_image-right_image_warped)**2, axis=2)<=epsilon**2, inside_border))
    return mask


def RANSAC_plane_depth(x, depth_known_finite, alpha, iter, p):
    theta = alpha * np.median(depth_known_finite)  # threshold to get inliers
    N = 3  # sampling number for model fitting
    num_points = len(depth_known_finite)  # total number of points to fit
    # store inliers information
    inliers = []
    inliers_count = 0
    inliers_ratio = 0
    x = np.asarray(x)
    x_hom = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0) #shape(3, num_points)

    for i in range(iter):
        flag = True
        while flag:
            samples = np.random.choice(num_points, size=N, replace=False)
            x_sample = x_hom[:, samples]
            depth_sample = depth_known_finite[samples]
            if np.linalg.matrix_rank(x_sample) == N:
                flag = False

        plane_tmp = normal_equation(x_sample.T, depth_sample)
        inliers_tmp = (x_hom.T.dot(plane_tmp)-depth_known_finite) <= theta
        inliers_tmp_count = np.sum(inliers_tmp)

        if inliers_tmp_count > inliers_count:
            inliers_count = inliers_tmp_count
            inliers_ratio = inliers_tmp_count/num_points
            inliers = np.where(inliers_tmp==True)[0]
            if len(inliers) != inliers_count:
                raise RuntimeError('Implementation error!')

        p_bound = 1 - (1 - inliers_ratio**N)**(i+1)
        if p_bound >= p:
            break
    plane_finite = normal_equation(x_hom[:, inliers].T, depth_known_finite[inliers])
    return plane_finite, inliers


## TODO: this is to be modified
def guided_filter(t, I, window_size, mu):
    """
    imguidedfilter with arguments 'NeighborhoodSize', 'DegreeOfSmoothing'
    :param t:
    :param I:
    :param window_size:
    :param mu:
    :return:
    """
    # t_refined = clip_to_unit_range(cv2.ximgproc.guidedFilter(I, t, window_size, mu))
    t_refined = imguidedfilter(t, I, (window_size, window_size), mu)
    return t_refined


def transmission_homogeneous_medium(d, beta, camera_path):
    """
    Calculate transmission ratio t in scattering model
    """
    l = distance_in_meter(d, camera_path)
    t = np.exp(-beta*l)
    return t


def transmission_postprocessing(t, I):
    window_size = 41
    mu = 1e-2
    t = clip_to_unit_range(guided_filter(t, I, window_size, mu))
    return t


def brightest_pixels_count(num_pixels, fraction):
    tmp = math.floor(fraction*num_pixels)
    return tmp+((tmp+1) % 2)


def get_dark_channel(image, window):
    # erode the image
    kernel = np.ones((window, window), np.uint8)
    image_erode = cv2.erode(image, kernel, iterations=1)
    dark_channel = np.min(image_erode, axis=-1)
    return dark_channel


def get_atmosphere_light(dark_channel, image):

    # Determine the number of brightest pixels in dark channel
    brightest_pixels_frac = 1e-3
    H, W = dark_channel.shape
    num_pixels = H*W
    brightest_pixels_num = brightest_pixels_count(num_pixels, brightest_pixels_frac)

    # get the indices of brightest pixels in dark channel
    sort_idx = np.argsort(np.ndarray.flatten(dark_channel))[::-1]
    brightest_pixels_idx = sort_idx[:brightest_pixels_num]

    gray_image = rgb2gray(image)
    gray_brightest_pixels = np.ndarray.flatten(gray_image)[brightest_pixels_idx]

    gray_median_intensity = np.median(gray_brightest_pixels)
    temp_idx = np.where(gray_brightest_pixels==gray_median_intensity)[0][0]
    x, y = np.unravel_index(brightest_pixels_idx[temp_idx], (H, W))

    L = image[x, y, :]
    return L, brightest_pixels_idx[temp_idx]


def generate_haze(image, L, t):
    L_map = np.repeat(np.repeat(L[np.newaxis, :], 2048, 0)[np.newaxis,...], 1024, 0)
    t_map = np.repeat(t[...,np.newaxis], 3, axis=2)
    hazy_image = image*t_map + L_map*(1-t_map)
    return hazy_image


