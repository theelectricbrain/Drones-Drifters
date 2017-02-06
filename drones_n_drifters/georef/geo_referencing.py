#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import numpy as np
# from pyproj import Proj, pj_list, pj_ellps
import cv2


def geo_ref_tracks(tracks, frame, uav, debug=False):
    """
    Geo-references tracks'points
    :param tracks: list of drifters' trajectories
    :param frame: CV2 frame
    :param uav: UAV class object
    :return: geo-referenced tracks in degrees and tracks relative to center point in meters
    """
    # Meter per pixel ratio
    # TODO: Lens correction could be needed here
    diagLength = 2.0 * np.tan(np.deg2rad(uav.FOV/2.0)) * uav.altitude
    nx = float(frame.shape[1])
    ny = float(frame.shape[0])
    phi = np.arctan(ny / nx)
    horiMpP = diagLength * np.cos(phi) / nx  # horizontal meters per pixel ratio
    vertiMpP = diagLength * np.sin(phi) / ny  # vertical meters per pixel ratio.
    if uav.yaw < 0.0:  # UAV convention
        alibi = True
    else:
        alibi = False
    yaw = np.abs(np.deg2rad(uav.yaw))
    # Need this before of tuples
    tracksInDeg = []
    tracksInRelativeM = []
    for tr in tracks:
        tracksInDeg.append([])
        tracksInRelativeM.append([])
    #  Relative distance
    for tr, TR in zip(tracks, tracksInRelativeM):
        for pt in tr:
            pt = list(pt)
            x = (pt[0] - (nx/2.0)) * horiMpP
            y = ((ny - pt[1]) - (ny/2.0)) * vertiMpP  # Origin frame is top left corner
            if alibi:
               # Correction with Active (aka Alibi) transformation
               xr = x * np.cos(yaw) - y * np.sin(yaw)
               yr = x * np.sin(yaw) + y * np.cos(yaw)
            else:
               # Correction with Passive (aka Alias) transformation
               xr = x*np.cos(yaw) + y*np.sin(yaw)
               yr = y*np.cos(yaw) - x*np.sin(yaw)
            TR.append([xr, yr])
    #  Conversion deg. to m. / Version 2.0
    y2lat = 1.0 / (110.54 * 1000.0)
    x2lon = 1.0 / (111.320 * 1000.0 * np.cos(np.deg2rad(uav.centreCoordinates[1])))
    lonC, latC = uav.centreCoordinates[0], uav.centreCoordinates[1]
    for tr, trM in zip(tracksInDeg, tracksInRelativeM):
        for ptM in trM:
            lon, lat = lonC + (ptM[0] * x2lon), latC + (ptM[1] * y2lat)
            tr.append([lon, lat])

    #  Conversion deg. to m. / version 1.0
    # proj = raw_input("Use default projection UTM/WGS84 (yes/no)?: ").upper()
    # if proj in "YES":
    #     myproj = Proj(proj='utm', ellps='WGS84')  # LatLon with WGS84 datum used by GPS units
    # else:
    #     print "Choose a coordinate projection from the following list:"
    #     for key in pj_list:
    #         print key + ": " + pj_list[key]
    #     proj = raw_input("Type in the coordinate projection: ")
    #     print "Choose a coordinate ellipse from the following list:"
    #     for key in pj_list:
    #         print key + ": " + pj_list[key]
    #     ellps = raw_input("Type in the coordinate ellipse: ")
    #     myproj = Proj(proj=proj, ellps=ellps)
    # xc, yc = myproj(uav.centreCoordinates[0], uav.centreCoordinates[1])
    # #  Absolute distance and conversion m. to deg.
    # for tr, trM in zip(tracksInDeg, tracksInRelativeM):
    #     for ptM in trM:
    #         x, y = xc + ptM[0], yc + ptM[1]
    #         lon, lat = myproj(x, y, inverse=True)
    #         tr.append([lon, lat])
    # #  Recompute relative distance in new referential
    # tracksInRelativeM = []
    # for tr in tracks:
    #     tracksInRelativeM.append([])
    # lat2m = 110.54 * 1000.0
    # lon2m = 111.320 * 1000.0 * np.cos(np.deg2rad(uav.centreCoordinates[1]))
    # for tr, trM in zip(tracksInDeg, tracksInRelativeM):
    #     for pt in tr:
    #         x = lon2m * (pt[0] - uav.centreCoordinates[0])
    #         y = lat2m * (pt[1] - uav.centreCoordinates[1])
    #         trM.append([x, y])

    return tracksInDeg, tracksInRelativeM

# TODO: def geo_ref_contours
def geo_ref_contours(surfTurbArea, uav, debug=False):
    """
    Geo-references surface turbulence areas

    :param surfTurbArea: frame of surface turbulence areas
    :param uav: UAV object
    :return: geo-referenced contours
    """
    # Find contours from white areas
    imgray = cv2.cvtColor(surfTurbArea,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        im = cv2.drawContours(surfTurbArea, contours, -1, (0,255,0), 3)
        cv2.namedWindow('Areas & contours', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Areas & contours', 1200, 1200)
        cv2.imshow('Areas & contours', im)
    # Reformating
    contoursList = []
    for cnt in contours:
        coordsList = []
        for coords in cnt:
            coordsList.append(tuple(coords[0]))
        contoursList.append(coordsList)
    # Georeference contours
    contoursInDeg, contoursInM = geo_ref_tracks(contoursList, surfTurbArea, uav, debug=debug)

    return contoursInDeg
