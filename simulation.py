import numpy as np
import cv2
import os
import pdb


from utils import *

image_list = ['aachen_000001_000019',
              'cologne_000088_000019',
              'dusseldorf_000020_000019',
              'hamburg_000000_046078']
image_subdir = ['camera', 'disparity', 'leftImg8bit', 'rightImg8bit']
image_basename = image_list[-1]

## Hyper-parameters
BETA_RGB = 0.01
WINDOW_SIZE = 15


# read images
print('Loading images...')
left_image_dir = os.path.join('data', image_subdir[2], '%s_%s.png' % (image_basename, image_subdir[2]))
right_image_dir = os.path.join('data', image_subdir[-1], '%s_%s.png' % (image_basename, image_subdir[-1]))
disparity_dir = os.path.join('data', image_subdir[1], '%s_%s.png' % (image_basename, image_subdir[1]))
camera_parameters_dir = os.path.join('data', image_subdir[0], '%s_%s.json' % (image_basename, image_subdir[0]))

left_image_uint8 = cv2.cvtColor(cv2.imread(left_image_dir), cv2.COLOR_BGR2RGB)
left_image = (left_image_uint8-left_image_uint8.min())/float(left_image_uint8.max()-left_image_uint8.min())
right_image_uint8 = cv2.cvtColor(cv2.imread(right_image_dir), cv2.COLOR_BGR2RGB)
right_image = (right_image_uint8-right_image_uint8.min())/float(right_image_uint8.max()-right_image_uint8.min())
left_disparity = cv2.imread(disparity_dir, cv2.IMREAD_UNCHANGED)

# step2: depth map calculation, denoising
print('Calculating depth map...')
depth_map, _ = depth_inpainting(left_disparity, camera_parameters_dir, left_image,
                             left_image_uint8, right_image)

inf_mask = depth_map==INF
depth_map[inf_mask] = 1000

# step3: transmission calculation
print('Calculating transmission ratio')
t_initial = transmission_homogeneous_medium(depth_map, BETA_RGB, camera_parameters_dir)

# step4: refine transmission
t = transmission_postprocessing(t_initial, left_image)

# step5: dark channel
left_image_dark = get_dark_channel(left_image, WINDOW_SIZE)

# step6: estimate atmospheric light
print('Estimating armospheric light...')
atmos_light, _ = get_atmosphere_light(left_image_dark, left_image)

# step7: generate haze
hazy_image = generate_haze(left_image, atmos_light, t)

plt.imshow(hazy_image)
plt.show()