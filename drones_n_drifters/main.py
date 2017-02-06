#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import sys
import cv2
import numpy as np
from progressbar import ProgressBar
from imgprocess.image_filtering import *
from imgprocess.image_post_processing import *
from traj.motion_tracking import *
from misc.utilities import *
from misc.read_input_files import *
from georef.geo_referencing import *
from oceano.oceanographic_quantities import *
from misc.write_files import *
# Quick fix
from skvideo.io import VideoCapture

### Exportation info ###
kmlName = "/home/grumpynounours/Desktop/test.kml"
matName = "/home/grumpynounours/Desktop/test_pyseidon_drifter.mat"
shpName = "/home/grumpynounours/Desktop/test.shp"

# Debug flag
debug = True

### Syncronisation and video splitting block ###
# TODO: to be developed
# UAV, CAP classes will be defined by synchronisation of the LOG class

# UAV attributes and parameters
#  TODO: UAV Attributes to retrieve from log.
uav = UAV("test_file.log", debug=debug)
#  Manual attributes definition. THis step will be looped over the video files trested
uav.centreCoordinates = (-66.3406, 44.2564557)  # (lon., lat.) in decimal degrees. Convention
uav.yaw = -38.2  # in degrees. Convention?
uav.FOV = 94.0  # in deg.
uav.altitude = 111.4  # in meters (feet to meter conversion here). Convention?
uav.timeRef = datetime(2016, 12, 01)

# Video capture
#   Manual defined. THis step will be looped
cap = CAP("/home/grumpynounours/Desktop/Electric_Brain/measurements/pumkin_passing_test.MOV")

### Video processing block ###
# Save Frames
saveFrames = False

# Color detection attributes
colorDetect = False  # perform color detection yes/no, true/false
#  HSV range value for Orange pumkin
# TODO: shall be turned into input in interface module
colorBounds = ([110, 50, 70], [135, 250, 230])  # shades of orange.
# For some reason the Hue value are inversed. When opening frame.jpg in GIMP, orange = light blue


# White patches masking attributes
maskOutWhitePatches = True
#  HSV range value for white surface waves
whiteBounds = ([0, 0, 235], [10, 20, 255])
# Version 1
#  Dilation kernel
# dilatationKernelSize = 401

# Circle detection...not working yet
circleDetection = False
#  circle sizes' range
minArea = 2
maxArea = 40

# Surface turbulence detection
surfaceTurbulenceDetection = True

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
iterationId = -1
pbar = ProgressBar()
for frame_id in pbar(range(cap.nbFrames)):
    ret, frame = cap.read()
    if ret:
        iterationId += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ## Detection ##
        # Color detection, here pumkin orange
        if colorDetect:
            greyScaleMask = color_detection(frame, colorBounds=colorBounds, debug=debug)
        else:
            greyScaleMask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Shape detection
        if circleDetection:
            circleMask = circle_detection(rgb, minArea, maxArea, debug=debug)
            # Resulting mask
            greyScaleMask = cv2.bitwise_and(greyScaleMask, greyScaleMask, mask=circleMask)

        # Surface Turbulence Detection
        if surfaceTurbulenceDetection:
            if iterationId == 0:
                surfTurbMask = 255 * np.ones((frame.shape[0], frame.shape[1], 3), np.uint8)
        ## Filtering ##
        # Mask-out white
        if maskOutWhitePatches or surfaceTurbulenceDetection:
            # Version 1
            # dilatedMaskWhite = white_patches_masking(rgb, whiteBounds=whiteBounds,
            #                                          dilatationKernelSize=dilatationKernelSize,
            #                                          debug=debug)
            # Version 2
            dilatedMaskWhite = white_patches_masking(rgb, whiteBounds=whiteBounds, debug=debug)
            if maskOutWhitePatches:
                # Resulting mask
                greyScaleMask = cv2.bitwise_and(greyScaleMask, greyScaleMask, mask=dilatedMaskWhite)
            if surfaceTurbulenceDetection:
                # Resulting surf. turb. mask
                surfTurbMask = cv2.bitwise_and(surfTurbMask, surfTurbMask, mask=dilatedMaskWhite)

        ## Motion tracking ##
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
tracksInDeg, tracksInRelativeM = geo_ref_tracks(tracks, frame, uav, debug=False)

### Compute flow velocities ###
d = velocities_from_geotracks(uav, cap, tracksInDeg, tracksInRelativeM, frameIdx, rw='1S', debug=debug)

### Surface Turbulence post-processing  and geo-referencing###
if surfaceTurbulenceDetection:
    surfTurbArea = surface_turbulence_area_post_process(surfTurbMask, debug=debug)
    surf_contours = geo_ref_contours(surfTurbArea, uav, debug=debug)

### Exportation block ###
# Export to kmz
write_tracks2kml(tracksInDeg, kmlName)
if surfaceTurbulenceDetection:
    kmlName[:-4] + "_surface_turbulence_areas.kml"
    write_contours2kml(surf_contours, kmlName)

# Export to matlab (based on drifters' file format)
write2drifter(d, uav, matName)

#Export to shapefile
write2shp(d, shpName)
# TODO: if surfaceTurbulenceDetection: write_contours2shp(surf_contours, fill=True)