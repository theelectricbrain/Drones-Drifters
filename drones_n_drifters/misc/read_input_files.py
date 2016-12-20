#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
from skvideo.io import VideoCapture
import numpy as np


class UAV:
    def __init__(self, logFileName, debug=False):
        self._debug = debug
        # Bunch of metadata related to the drone itself
        self.model = raw_input("What is the UAV's model type (Phantom 3, Iris,...)?: ").upper().replace(" ", "")
        # FOV (Field Of View). Assuming automatic vertical alignment of the camera
        if "PHANTOM3" in self.model:
            #self.vertiFOV = 61.9  # in degrees
            #self.horiFOV = 82.4  # in degrees
            self.FOV = 94.0  # in degrees
        # To be defined once reading through log and video
        self.centreCoordinates = (None, None)  # (lon., lat.) in decimal degrees. Convention
        #  Convention NED, North-East-Down, X-Y-Z, Roll-Pitch-Yaw.
        self.roll = None  # longitudinal axis rotation in degrees. Convention?
        self.pitch = None  # lateral axis rotation in degrees. Convention?
        self.yaw = None  # vertical axis rotation in degrees. North = 0deg. Positive clockwise.
        #  Not sure which convention here!
        self.altitude = None # in meters. from ground up
        # Reference time of the log
        self.timeRef = None  # in datetime

class CAP(VideoCapture):
    def __init__(self, videoFileName, debug=False):
        # cv2.VideoCapture.__init__(self, videoFileName)
        # Does not work on my machine! \Quick fix using VideoCapture from skvideo
        VideoCapture.__init__(self, videoFileName)
        # Sub-class attributes
        self._debug = debug
        self.videoFileName = videoFileName
        self.nbFrames = self.info['streams'][0]['nb_frames']
        self.fps = int(np.round(self.nbFrames/float(self.info['format']['duration'])))
        # self.timeRef = None  # in datetime