#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division

class UAV:
    def __init__(self, logFileName, debug=False):

        # Bunch of metadata related to the drone itself
        self.model = raw_input("What is the UAV's model type (Phantom 3, Iris,...)?: ").upper().replace(" ", "")
        # FOV (Field Of View). Assuming automatic vertical alignment of the camera
        if "PHANTOM3" in self.model:
            self.vertiFOV = 61.9  # in degrees
            self.horiFOV = 82.4  # in degrees
        # To be defined once reading through log and video
        self.centreCoordinates = (None,None)  # (lon., lat.) in decimal degrees. Convention
        #  Convention NED, North-East-Down, X-Y-Z, Roll-Pitch-Yaw.
        self.roll = None  # longitudinal axis rotation in degrees. Convention?
        self.pitch = None  # lateral axis rotation in degrees. Convention?
        self.yaw = None  # vertical axis rotation in degrees. North = 0deg. Convention?
        #
        self.altitude = None # in meters. from ground up

