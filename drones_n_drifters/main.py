#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import sys
import cv2
import numpy as np
from pyproj import Proj, pj_list
from imgprocess.image_filtering import *
from traj.motion_tracking import *
from misc.utilities import *
from misc.read_input_files import *
from georef.geo_referencing import *
from oceano.oceanographic_quantities import *
# Quick fix
from skvideo.io import VideoCapture

### Syncronisation and video splitting block ###
# TODO: to be developed
# UAV, CAP classes will be defined by synchronisation of the LOG class

### Video processing block ###
# Debug flag
debug=True

# Save Frames
saveFrames = False

# UAV attributes and parameters
#  TODO: UAV Attributes to retrieve from log.
uav = UAV("test_file.log", debug=debug)
#  Manual attributes definition. THis step will be looped
uav.centreCoordinates = (-66.33558489, 44.28223669) # (lon., lat.) in decimal degrees. Convention
uav.yaw = np.deg2rad(10.0)  # in radian. Convention?
uav.vertiFOV = np.deg2rad(61.9)  # in rad.
uav.horiFOV = np.deg2rad(82.4)  # in rad.
uav.altitude = 300.0 / 3.28 # in meters (feet to meter conversion here). Convention?
#uav.timeRef = datetime(2016, 12, 01)

# Video capture
#   Manual defined. THis step will be looped
cap = CAP("/home/grumpynounours/Desktop/Electric_Brain/measurements/pumkin_passing_test.MOV")

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
frameIdx = []
for frame_id in range(cap.nbFrames):
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
            tracks, frameIdx = motion_tracking_Lucas_Kanade(tracks, frameIdx, frame_id, prev_gray, greyScaleMask,
                                                         minTrackLength=minTrackLength,
                                                         detect_interval=detect_interval,
                                                         lk_params=lk_params,
                                                         feature_params=feature_params,
                                                         debug=debug)
        else:
            #TODO implement alternative motion tracker
            print("Option not supported yet. Alternative motion tracker needed:", sys.exc_info()[0])
            raise
        # Incrementation
        prev_gray = greyScaleMask

        # Breaking for loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print "Breaking loop"
            break
        # Save Video
        if saveFrames:
            cv2.imwrite("image"+str(frame_id).zfill(4)+".png", rgb)

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

# Turn tracks and frameIdx list into array
tracks = np.asarray(tracks)
frameIdx = np.asarray(frameIdx)

# Filtering and selecting trajectories
#  Selection through trajectories' sizes histogram
tracks, frameIdx = tracks_sizes_selection(tracks, frameIdx, rgb)
# TODO: Code interface/user-input based tracks selection

### Geo-referencing block ###
tracks, tracksInM = geo_ref_tracks(tracks, frame, uav, debug=False)

### Compute flow velocities ###
d = velocities_from_geotracks(uav, cap, tracks, tracksInM, frameIdx, rw='1S', debug=debug)

### Exportation block ###
# TODO: Use same format as in/home/grumpynounours/Desktop/Github/PySeidon_dvt/data4tutorial/drifter_GP_01aug2013.mat and drifterclass
