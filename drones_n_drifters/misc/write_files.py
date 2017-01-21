#!/usr/bin/python2.7
# encoding: utf-8

import numpy as np
import simplekml
from scipy.io import savemat
from utilities import *

def write2kml(tracksInDeg, kmlName):
    """
    Writes trajectories in KML file

    :param tracksInDeg: trajectories in degrees
    :param kmlName: KML file name
    """
    kml = simplekml.Kml()
    for ii, tr in enumerate(tracksInDeg):
        lin = kml.newlinestring(name="Trajectory "+str(ii), coords=tr)
    kml.save(kmlName)

def write2drifter(d, uav, matName):
    """
    Writes oceanographic quantities to matfile

    :param d: dataframe
    :param uav: UAV object
    :param matName: mat filename
    """
    d4mat = {}
    d4mat['comments'] = ["trajectories from pumpkins"]
    d4mat['comments'].append("reference time: " + uav.timeRef.strftime("%Y-%m-%d %H:%M:%S"))
    u = []
    v = []
    lon = []
    lat = []
    times = []
    # TODO: unknown bug - u and v inversed !!!
    for key in d.keys():
        u.extend(d[key]['U'].tolist())
        v.extend(d[key]['V'].tolist())
        lon.extend(d[key]['longitude'].tolist())
        lat.extend(d[key]['latitude'].tolist())
        times.extend(datetime_to_mattime(d[key].index.tolist()))
    d4mat['velocity'] = {}
    d4mat['velocity']['u'] = np.asarray(u)
    d4mat['velocity']['v'] = np.asarray(v)
    d4mat['velocity']['vel_lon'] = np.asarray(lon)
    d4mat['velocity']['vel_lat'] = np.asarray(lat)
    d4mat['velocity']['vel_time'] = np.asarray(times)

    savemat(matName, d4mat)