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
# Need this before of tuples
tempTracks = []
for tr in tracks:
    tempTracks.append([])
# TODO: to be developed
# TODO: Import Tracks in panda frame and start working on them (georef, resampling,...)
# Attributes to retrieve from log. TODO: define class UAV
centreCoordinates = (-66.33558489, 44.28223669) # (lon., lat.) in decimal degrees. Convention
yaw = np.deg2rad(10.0)  # in radian. Convention?
vertiFOV = np.deg2rad(61.9)  # in rad.
horiFOV = np.deg2rad(82.4)  # in rad.
altitude = 300.0 / 3.28 # in meters (feet to meter conversion here). Convention?
nx = float(frame.shape[0])
ny = float(frame.shape[1])
horiMpP = (2.0*np.tan(horiFOV/2.0))/nx  # horizontal meters per pixel ratio
vertiMpP = (2.0*np.tan(vertiFOV/2.0))/ny  # vertical meters per pixel ratio, function of the altitude. Lens correction could be needed here
#  Relative distance correction with Passive (aka Alias) transformation
for tr, TR in zip(tracks, tempTracks):
    for pt in tr:
        pt = list(pt)
        x = pt[0] - (nx/2.0)
        y = pt[1] - (ny/2.0)
        xr = x*np.cos(yaw) + y*np.sin(yaw)
        yr = y*np.cos(yaw) - x*np.sin(yaw)
        TR.append([xr, yr])
#  Conversion deg. to m.
proj = raw_input("Is the projection UTM (yes/no)?: ").upper()
if proj in "YES":
    proj = 'utm'
else:
    print "Choose a coordinate projection from the following list:"
    for key in pj_list:
        print key + ": " + pj_list[key]
    proj = raw_input("Type in the coordinate projection: ")
myproj = Proj(proj=proj)
xc, yc = myproj(centreCoordinates[0], centreCoordinates[1])
#  Absolute distance and conversion m. to deg.
for tr in tempTracks:
    for pt in tr:
        lon, lat = myproj(xc + pt[0], yc + pt[1], inverse=True)
        pt[0] = lon
        pt[1] = lat
# Need this before of tuples
tracks = []
for tr in tempTracks:
    tracks.append([])
for tr, TR in zip(tempTracks, tracks):
    for pt in tr:
        TR.append(tuple(pt))
del tempTracks

### Exportation block ###
# TODO: to be developed
