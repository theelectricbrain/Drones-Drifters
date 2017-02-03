#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division

import cv2
import numpy as np


def surface_turbulence_area_post_process(surfTurbMask, debug=False):
    """
    Inverses, erodes, dilates and smooths surface turbulence areas
    :param surfTurbMask: surface turbulence mask
    :return: surface turbulence area
    """
    surfTurbArea = 255 - surfTurbMask.copy()  # inverse black and white
    kernel = np.ones((5, 5), np.uint8)
    surfTurbArea = cv2.erode(surfTurbArea, kernel, iterations=20)  # erodes
    surfTurbArea = cv2.dilate(surfTurbArea, kernel, iterations=20)  # dilates
    surfTurbArea = cv2.GaussianBlur(surfTurbArea, (51, 51), 0)  # smooths edges

    if debug:
        cv2.namedWindow('surface turbulence area', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('surface turbulence area', 1200, 1200)
        cv2.imshow('surface turbulence area', surfTurbArea)

    return surfTurbArea

