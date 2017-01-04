#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import numpy as np
from pyproj import Proj, pj_list, pj_ellps


def geo_ref_tracks(tracks, frame, UAV, debug=False):
    """
    Geo-references tracks'points
    :param tracks: list of drifters' trajectories
    :param frame: CV2 frame
    :param UAV: UAV class object
    :return: geo-referenced tracks in degrees and meters
    """
    # Meter per pixel ratio
    # TODO: Lens correction could be needed here
    diagLength = 2.0 * np.tan(np.deg2rad(UAV.FOV/2.0)) * UAV.altitude
    nx = float(frame.shape[1])
    ny = float(frame.shape[0])
    phi = np.arctan(ny / nx)
    horiMpP = diagLength * np.cos(phi) / nx  # horizontal meters per pixel ratio
    vertiMpP = diagLength * np.sin(phi) / ny  # vertical meters per pixel ratio.
    if UAV.yaw > 0.0:  # UAV convention
        alibi = True
    else:
        alibi = False
    yaw = np.abs(np.deg2rad(UAV.yaw))
    # Need this before of tuples
    tempTracksInDeg = []
    tempTracksInM = []
    for tr in tracks:
        tempTracksInDeg.append([])
        tempTracksInM.append([])
    #  Relative distance
    for tr, TR in zip(tracks, tempTracksInM):
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
    #  Conversion deg. to m.
    proj = raw_input("Use default projection UTM/WGS84 (yes/no)?: ").upper()
    if proj in "YES":
        myproj = Proj(proj='utm', ellps='WGS84')  # LatLon with WGS84 datum used by GPS units
    else:
        print "Choose a coordinate projection from the following list:"
        for key in pj_list:
            print key + ": " + pj_list[key]
        proj = raw_input("Type in the coordinate projection: ")
        print "Choose a coordinate ellipse from the following list:"
        for key in pj_list:
            print key + ": " + pj_list[key]
        ellps = raw_input("Type in the coordinate ellipse: ")
        myproj = Proj(proj=proj, ellps=ellps)
    xc, yc = myproj(UAV.centreCoordinates[0], UAV.centreCoordinates[1])
    #  Absolute distance and conversion m. to deg.
    for tr, trM in zip(tempTracksInDeg, tempTracksInM):
        for ptM in trM:
            x, y = xc + ptM[0], yc + ptM[1]
            lon, lat = myproj(x, y, inverse=True)
            tr.append([lon, lat])
    # Need this before of tuples
    tracksInDeg = []
    tracksInM = []
    for tr in tempTracksInDeg:
        tracksInDeg.append([])
        tracksInM.append([])
    for tr, TR, trM, TRM in zip(tempTracksInDeg, tracksInDeg, tempTracksInM, tracksInM):
        for pt, ptM in zip(tr, trM):
            TR.append(tuple(pt))
            TRM.append(tuple(ptM))
    #del tempTracksInDeg, tempTracksInM

    return tracksInDeg, tracksInM