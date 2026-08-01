# """ 
# My CAR CONFIG 

# This file is read by your car application's manage.py script to change the car
# performance

# If desired, all config overrides can be specified here.
# The update operation will not touch this file.
# """

# Example: enable the "all_conditions" lighting-robustness augmentation
# profile (recommended when training one model on combined day/midday/
# night data). Uncomment to use; see cfg_complete.py for what each
# AUG_* setting does and other example profiles (indoor/outdoor/low_light).
# AUGMENTATIONS = ['BRIGHTNESS', 'BLUR', 'SHADOW', 'GAMMA', 'NOISE']
# AUG_GAMMA_RANGE = (60, 160)
# AUG_NOISE_PROBABILITY = 0.3

# Match the resolution the camera actually records at. The tubs in
# data_baseline/ are 192x108; the stock defaults in cfg_complete.py are
# 160x120, and load_image_sized() (donkeycar/utils.py) silently resizes any
# frame that doesn't match. Leaving the defaults in place therefore squeezed
# every 192x108 frame into 160x120 - a horizontal squash plus a vertical
# stretch - so the model trained on a distorted aspect ratio that the live
# camera never produces. Setting these makes training and driving agree.
IMAGE_W = 192
IMAGE_H = 108

# Mask off the top of the frame so the model can only learn from the track,
# not from background landmarks. Background features are the easiest signal
# available during training, so a model that can see them will use them -
# and then fails when a different time of day makes them look foreign.
# Removing them from the input is what makes the AUGMENTATIONS above worth
# applying. On this track the upper frame is buildings, windows, doorways
# and sky, all of which change completely between morning and evening.
#
# This is a mask, not a resize: ImgCropMask zeroes the region and returns
# the same IMAGE_H x IMAGE_W array, so IMAGE_H/IMAGE_W and the camera
# resolution stay untouched. Because TRANSFORMATIONS are applied during
# training AND while driving, this same setting must be present in the
# myconfig.py used for training and the one on the car - a mismatch is
# silent, and shows up only as a car that steers badly.
#
# POST_ rather than plain TRANSFORMATIONS: training applies
# TRANSFORMATIONS -> AUGMENTATIONS -> POST_TRANSFORMATIONS, and BRIGHTNESS
# / NOISE would lift the masked region off zero if the mask ran first,
# leaving training frames whose top half doesn't match the pure black the
# model sees when driving. Cropping last keeps both paths identical.
#
# 55 of 108 rows masks the top 51% of the frame. This is deeper than the 45
# used earlier: previewing LANE_ISOLATE on data_baseline showed that wall
# rails, window frames and doorways are thin bright structures too, so they
# survive that transform looking much like lane tape, and 45 left a band of
# them in the frame whenever the camera pitched up. Held constant across all
# trials so augmentation and transform effects stay comparable.
POST_TRANSFORMATIONS = []
ROI_CROP_TOP = 55

# The OAK-D part returns getCvFrame(), which is BGR, and nothing converts it
# before the tub writes it via PIL (which assumes RGB) - so the recorded
# JPEGs hold BGR data in an RGB container, and the genuinely yellow centre
# tape reads back as cyan. This is self-consistent for the model (the same
# swap applies when driving), but LANE_ISOLATE's chroma channel has to know
# the real channel order to target the real hue of the tape.
LANE_ISOLATE_COLOR_ORDER = 'bgr'
LANE_ISOLATE_CHROMA_ANGLE = 90.0   # yellow; measured a*=-1, b*=+33
ROI_CROP_BOTTOM = 0
ROI_CROP_LEFT = 0
ROI_CROP_RIGHT = 0

