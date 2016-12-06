#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import numpy as np
from pyproj import Proj, pj_list


def geo_ref_tracks(tracks, frame, UAV, debug=False):
    """
    Geo-references tracks'points
    :param tracks: list of drifters' trajectories
    :param frame: CV2 frame
    :param UAV: UAV class object
    :return: geo-referenced tracks
    """
    # Need this before of tuples
    tempTracks = []
    for tr in tracks:
        tempTracks.append([])
    nx = float(frame.shape[0])
    ny = float(frame.shape[1])
    horiMpP = (2.0*np.tan(UAV.horiFOV/2.0)*UAV.altitude)/nx  # horizontal meters per pixel ratio
    vertiMpP = (2.0*np.tan(UAV.vertiFOV/2.0)*UAV.altitude)/ny  # vertical meters per pixel ratio, function of the altitude. Lens correction could be needed here
    #  Relative distance correction with Passive (aka Alias) transformation
    for tr, TR in zip(tracks, tempTracks):
        for pt in tr:
            pt = list(pt)
            x = (pt[0] - (nx/2.0)) * horiMpP
            y = (pt[1] - (ny/2.0)) * vertiMpP
            xr = x*np.cos(UAV.yaw) + y*np.sin(UAV.yaw)
            yr = y*np.cos(UAV.yaw) - x*np.sin(UAV.yaw)
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
    xc, yc = myproj(UAV.centreCoordinates[0], UAV.centreCoordinates[1])
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

    return tracks