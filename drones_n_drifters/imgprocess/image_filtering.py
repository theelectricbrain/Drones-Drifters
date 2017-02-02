#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division

import cv2
import skvideo.io
import numpy as np

# def color_detection(rgb, colorBounds=([180, 69, 0], [240, 200, 240]), debug=False):
#     """
#     Detects given colors and produces grey mask accordingly
#     :param rgb: RGB frame
#     :param colorBounds: detection's color bounds, (low, high) = ([r,g,b], [R,G,B])
#     :return: grey scale image
#     """
#     # Color detection
#     lower = np.array(colorBounds[0], dtype="uint8")
#     upper = np.array(colorBounds[1], dtype="uint8")
#     greyMask = cv2.inRange(rgb, lower, upper)
#     if debug:
#         cv2.namedWindow('color detection', cv2.WINDOW_NORMAL)
#         cv2.resizeWindow('color detection', 1200, 1200)
#         cv2.imshow('color detection', greyMask)
#
#     return greyMask

def color_detection(frame, colorBounds=([180, 69, 0], [240, 200, 240]), debug=False):
    """
    Detects given colors and produces grey mask accordingly
    :param frame: BGR frame
    :param colorBounds: detection's color bounds, (low, high) = ([h,s,v], [H,S,V])
    :return: grey scale image
    """
    # Conversion RGB to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of detected color in HSV
    lower_color = np.array(colorBounds[0])
    upper_color = np.array(colorBounds[1])
    # Threshold the HSV image to get only specified colors
    greyMask = cv2.inRange(hsv, lower_color, upper_color)

    if debug:
        cv2.namedWindow('color detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('color detection', 1200, 1200)
        cv2.imshow('color detection', greyMask)

    return greyMask

def white_patches_masking(frame, whiteBounds = ([235, 235, 235], [255, 255, 255]),
                          dilatationKernelSize=301, debug=False):
    """
    Detection white colored pixels, dilates them and returns resulting grey scale mask
    :param frame: HSV frame
    :param whiteBounds: bounds of the color white, (low, high) = ([r,g,b], [R,G,B])
    :param dilatationKernel: dilatation kernel, (nb x pixels, nb y pixels)
    :return: grey scale frame masking out white patches
    """
    dilatationKernel = np.ones((dilatationKernelSize, dilatationKernelSize))
    maskWhite = color_detection(frame, colorBounds=whiteBounds)
    # Dilate white patches
    dilatedMaskWhite = cv2.bitwise_not(cv2.dilate(maskWhite, dilatationKernel))
    # Check patches size and make sure that they are bigger than dilatation kernel (due to chaos white pixels)
    # They look like kernel sized squares
    circularity = 0.785  # circularity for a square
    minPixel = dilatationKernel.size  # single dilated white pixel
    maxPixel = np.round(minPixel * 1.2)
    keypoints = blob_detector(dilatedMaskWhite, minPixel, maxPixel, circularity)
    # setting black square to white
    for kp in keypoints:
        i = int(np.round(kp.pt[1]))
        j = int(np.round(kp.pt[0]))
        interval = int(np.round(np.round(dilatationKernelSize * 1.1 / 2.0)))  # interval = half kernel + 10%
        dilatedMaskWhite[(i - interval):(i + interval), (j - interval):(j + interval)] = 255
    if debug:
        cv2.namedWindow('white detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('white detection', 1200, 1200)
        cv2.namedWindow('white patches dilatation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('white patches dilatation', 1200, 1200)
        cv2.imshow('white detection', maskWhite)
        im_with_keypoints = cv2.drawKeypoints(dilatedMaskWhite, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('white patches dilatation', im_with_keypoints)

    return dilatedMaskWhite

# TODO: not working
def disk_filtering(rgb, minPixel, maxPixel, debug=False):
    """
    Detects disk/circle and masks the rest

    :param rgb: RGB frame
    :param minPixel: minimum pixel size of the disks
    :param maxPixel: maximum pixel size of the disks
    :return: grey mask where disks are white squares and the rest is black
    """
    # detect white shades
    inverseGrey = cv2.cvtColor(cv2.bitwise_not(rgb), cv2.COLOR_RGB2GRAY)
    circularity = 1.0  # = circle
    keypoints = blob_detector(inverseGrey, minPixel, maxPixel, circularity)
    # setting disks to white squares and rest to black
    diskMask = inverseGrey.copy()
    diskMask[:] = 0
    for kp in keypoints:
        i = int(np.round(kp.pt[1]))
        j = int(np.round(kp.pt[0]))
        diskMask = cv2.circle(diskMask, (i, j), 10, 255, -1)
    if debug:
        cv2.namedWindow('disk filtering', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('disk filtering', 1200, 1200)
        cv2.imshow('disk filtering', diskMask)

    return diskMask

def circle_detection(rgb, minArea=10, maxArea=30, minVertices=6, maxVertices=10, debug=False):
    """
    Detects circle of given pixel size range
    :param rgb: RGB frame
    :param minArea: minimum circles' pixel size
    :param maxArea: maximum circles' pixel size
    :param minVertices: minimum circles' vertex number
    :param maxVertices: maximum circles' vertex number
    :return: grey mask where circles are white squares and the rest is black
    """

    # ref. http://layer0.authentise.com/detecting-circular-shapes-using-contours.html

    # ENhanced edges
    bilateral_filtered_image = cv2.bilateralFilter(rgb, 5, 175, 175)
    # Detect edges
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
    # Find contours
    _, contours, _= cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours
    contour_list = []
    for contour in contours:
            approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            area = cv2.contourArea(contour)
            if (((area > minArea) & (area < maxArea)) &
               ((len(approx) > minVertices) & (len(approx) < maxVertices))):
                    contour_list.append(contour)
    # Create filter
    circleMask = cv2.drawContours(np.zeros(rgb.shape[:2]),contour_list,
                                  -1, 255, int(np.sqrt(maxArea)))

    if debug:
        cv2.namedWindow('circle detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('circle detection', 1200, 1200)
        rgbCircles = cv2.drawContours(rgb.copy(), contour_list,  -1, (255,0,0), 2)
        cv2.imshow('circle detection', rgbCircles)

    return circleMask

def blob_detector(gray, minPixel, maxPixel, circularity, debug=False):
    """
    Detects blobs of given size and circularity
    :param gray:
    :param minPixel:
    :param maxPixel:
    :param circularity:
    :param debug:
    :return:
    """

    # ref. https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = minPixel
    params.maxArea = maxPixel

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