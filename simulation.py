import numpy as np
import cv2
import os


image_list = ['aachen_000001_000019',
              'cologne_000088_000019',
              'dusseldorf_000020_000019',
              'hamburg_000000_046078']
image_subdir = ['camera', 'disparity', 'leftImg8bit', 'rightImg8bit']
image_basename = image_list[0]

# step1: read in images
left_image_dir = os.path.join('data', image_subdir[2], '%s_%s.png' % (image_list[0], image_subdir[2]))
right_image_dir = os.path.join('data', image_subdir[-1], '%s_%s.png' % (image_list[0], image_subdir[2]))
depth_map_dir = os.path.join('data', image_subdir[1], '%s_%s.png' % (image_list[0], image_subdir[2]))
camera_parameters_dir = os.path.join('data', image_subdir[0], '%s_%s.json' % (image_list[0], image_subdir[0]))

left_image_uint8 = cv2.cvtColor(cv2.imread(left_image_dir), cv2.COLOR_BGR2RGB)
left_image = (left_image_uint8-left_image_uint8.min())/float(left_image_uint8.max()-left_image_uint8.min())
right_image_uint8 = cv2.imread(right_image_dir)
left_depth_map = cv2.imread(depth_map_dir, cv2.IMREAD_UNCHANGED)

# step2: depth map calculation, denoising
beta_visible = 0.01
window_size = 15

# step3: transmission calculation


# step4: refine transmission

# step5: dark channel

# step6: estimate atmospheric light

# step7: generate haze