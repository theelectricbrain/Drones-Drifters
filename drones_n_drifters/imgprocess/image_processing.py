#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division

import cv2
import skvideo.io
import numpy as np

def color_detection(rgb, colorBounds=([180, 69, 0], [240, 200, 240]), debug=False):
    """
    Detects given colors and produces grey mask accordingly
    :param rgb: RGB frame
    :param colorBounds: detection's color bounds, (low, high) = ([r,g,b], [R,G,B])
    :return: grey scale image
    """
    if debug:
        cv2.namedWindow('color detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('color detection', 1200, 1200)
    # Color detection
    lower = np.array(colorBounds[0], dtype="uint8")
    upper = np.array(colorBounds[1], dtype="uint8")
    greyMask = cv2.inRange(rgb, lower, upper)
    if debug:
        cv2.imshow('color detection', greyMask)

    return greyMask

def white_patches_masking(rgb, whiteBounds = ([240, 235, 240], [255, 255, 255]),
                          dilatationKernelSize=301, debug=False):
    """
    Detection white colored pixels, dilates them and returns resulting grey scale mask
    :param rgb: RGB frame
    :param whiteBounds: bounds of the color white, (low, high) = ([r,g,b], [R,G,B])
    :param dilatationKernel: dilatation kernel, (nb x pixels, nb y pixels)
    :return: grey scale frame masking out white patches
    """
    if debug:
        cv2.namedWindow('white detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('white detection', 1200, 1200)
        cv2.namedWindow('white patches dilatation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('white patches dilatation', 1200, 1200)
    dilatationKernel = np.ones((dilatationKernelSize, dilatationKernelSize))
    maskWhite = color_detection(rgb, colorBounds=whiteBounds)
    # Dilate white patches
    dilatedMaskWhite = cv2.bitwise_not(cv2.dilate(maskWhite, dilatationKernel))
    # TODO: check patches size and make sure that they are bigger than dilatation kernel (due to chaos white pixels)
    #       they look like kernel sized squares
    circularity = 0.785  # circularity for a square
    minPixel = dilatationKernel.size  # single dilated white pixel
    keypoints = blob_detector(dilatedMaskWhite, minPixel, circularity)
    # setting black square to white
    for kp in keypoints:
        i = int(np.round(kp.pt[1]))
        j = int(np.round(kp.pt[0]))
        interval = int(np.round(np.round(dilatationKernelSize * 1.1 / 2.0)))  # interval = half kernel + 10%
        dilatedMaskWhite[(i - interval):(i + interval), (j - interval):(j + interval)] = 255
    if debug:
        cv2.imshow('white detection', maskWhite)
        im_with_keypoints = cv2.drawKeypoints(dilatedMaskWhite, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('white patches dilatation', im_with_keypoints)

    return dilatedMaskWhite

def blob_detector(gray, minPixel, circularity, debug=False):

    # ref. https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = minPixel
    params.maxArea = np.round(minPixel * 1.2)

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = circularity

    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87
    #
    # # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(gray)

    # # Draw detected blobs as red circles.
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # # the size of the circle corresponds to the size of blob
    #
    # im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # # Show blobs
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)

    return keypoints