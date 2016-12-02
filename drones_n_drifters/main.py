#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imgprocess.image_filtering import *
from traj.motion_tracking import *
from misc.utilities import *
# Quick fix
from skvideo.io import VideoCapture

### Syncronisation and video splitting block ###
# TODO: to be developed

### Video processing block ###
# Debug flag
debug=True

# Save Frames
saveFrames = False

# Video capture
# cap = cv2.VideoCapture("/home/grumpynounours/Desktop/Electric_Brain/measurements/pumkin_passing_cut.avi")
# quick fix
cap = VideoCapture("/home/grumpynounours/Desktop/Electric_Brain/measurements/pumkin_passing_test.MOV")

# Color detection attributes
colorDetect = False  # perform color detection yes/no, true/false
#  RBG range value for Orange pumkin
colorBounds = ([180, 60, 0], [240, 220, 250])  # shades of orange

# White patches masking attributes
maskOutWhitePatches = True
#  RBG range value for white surface waves
whiteBounds = ([245, 245, 245], [255, 255, 255])
#  Dilation kernel
dilatationKernelSize = 401

# Circle detection...not working yet
circleDetection = False
#  circle sizes' range
minArea = 2
maxArea = 40

# Motion tracking attributes
lk_params = dict(winSize=(15, 15),  # what are those?
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#  ref. http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
#  ref. http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
feature_params = dict(maxCorners=500,  # what are those?
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
#  ref. http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
#  ref. http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack
#  Minimum track length as storage criteria
minTrackLength = 200
detect_interval = 5  # Default value = 5.

# Initialise loop through video frames
tracks = []
frame_idx = 0
for l in range(cap.info['streams'][0]['nb_frames']):
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Color detection, here pumkin orange
        if colorDetect:
            greyScaleMask = color_detection(rgb, colorBounds=colorBounds, debug=debug)
        else:
            greyScaleMask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mask-out white
        if maskOutWhitePatches:
            dilatedMaskWhite = white_patches_masking(rgb, whiteBounds=whiteBounds,
                                                     dilatationKernelSize=dilatationKernelSize,
                                                     debug=debug)
            # Resulting mask
            greyScaleMask = cv2.bitwise_and(greyScaleMask, greyScaleMask, mask=dilatedMaskWhite)

        # Shape detection
        if circleDetection:
            circleMask = circle_detection(rgb, minArea, maxArea, debug=debug)
            # Resulting mask
            greyScaleMask = cv2.bitwise_and(greyScaleMask, greyScaleMask, mask=circleMask)

        # Motion tracking
        if len(tracks) == 0:
            prev_gray = greyScaleMask
        if not circleDetection:
            tracks = motion_tracking_Lucas_Kanade(tracks, frame_idx, prev_gray, greyScaleMask,
                                                  minTrackLength=minTrackLength, detect_interval=detect_interval,
                                                  lk_params=lk_params, feature_params=feature_params, debug=debug)
        else:
            #TODO implement alternative motion tracker
            print("Option not supported yet. Alternative motion tracker needed:", sys.exc_info()[0])
            raise
        # Incrementation
        frame_idx += 1
        prev_gray = greyScaleMask

        # Breaking for loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print "Breaking loop"
            break
        # Save Video
        if saveFrames:
            cv2.imwrite("image"+str(l).zfill(4)+".png", rgb)
# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

# Turn tracks list into array
tracks = np.asarray(tracks)

# Filtering and selecting trajectories
#  Selection through trajectories' sizes histogram
tracks = tracks_sizes_selection(tracks, rgb)
# TODO: Code interface/user-input based tracks selection

### Geo-referencing block ###
# TODO: to be developed
# TODO: Import Tracks in panda frame and start working on them (georef, resampling,...)

### Exportation block ###
# TODO: to be developed
