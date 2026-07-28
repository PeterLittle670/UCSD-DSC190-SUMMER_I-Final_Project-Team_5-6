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

# Mask off the top half of the frame so the model can only learn from the
# track, not from background landmarks. Background features are the easiest
# signal available during training, so a model that can see them will use
# them - and then fails when a different time of day makes them look
# foreign. Removing them from the input is what makes the AUGMENTATIONS
# above worth applying.
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
# On the cone-track sample in Data/cone_images_dc (measured at the 160x120
# the model actually sees), blue-cone pixels peak in rows 56-82 - the cones
# sit just above the point where the pavement disappears, because that band
# is where mid- and far-distance markers appear. Sky, building and trees are
# all above row ~52. Cropping at 55 keeps 81% of cone pixels; 60 keeps 68%,
# 65 keeps 50%, 70 keeps 30%. Erring shallow is deliberate: too little crop
# leaves some background the AUGMENTATIONS can still cover, while too much
# blinds the model to the far cones it steers by, and camera pitch and
# horizon height vary across sessions.
POST_TRANSFORMATIONS = ['CROP']
ROI_CROP_TOP = 55       # rows masked off the top (IMAGE_H = 120)
ROI_CROP_BOTTOM = 0
ROI_CROP_LEFT = 0
ROI_CROP_RIGHT = 0

