#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def velocities_from_geotracks(uav, cap, tracks, tracksInM, frameIdx, rw='1S', debug=False):
    """
    Computes flow velocities from geo-referenced drifter's trajectories
    :param uav: UAV object
    :param cap: CAP object
    :param tracks: Drifter's trajectories, in Deg.
    :param tracksInM: Drifter's trajectories, in m.
    :param frameIdx: Video frames' indexes
    :param rw: resampling window, '1S' = one second
    :return: dictionary of dataframes
    """
    uav.timeRef = datetime(2016, 12, 01)
    # Defining dict of dataframes
    d = {}
    for ii, tr, trM, fi in zip(range(len(tracks)), tracks, tracksInM, frameIdx):
        timeRef = []
        for idx in fi:
            timeRef.append(uav.timeRef + timedelta(seconds=idx * (1.0/cap.fps)))
        lons = []
        lats = []
        xs = []
        ys = []
        for pt, ptM in zip(tr, trM):
            lons.append(pt[0])
            lats.append(pt[1])
            xs.append(ptM[0])
            ys.append(ptM[1])
        d['track' + str(ii)] = pd.DataFrame({'longitude': lons, 'latitude': lats, 'x': xs, 'y': ys}, index=timeRef)
        # Resampling
        d['track' + str(ii)] = d['track' + str(ii)].resample(rw).mean()
    # Computing velocities
    for key in d.keys():
        d[key]['Time'] = d[key].index.asi8
        dist = d[key].diff().fillna(0.)
        dist['Dist'] = np.sqrt(dist.x ** 2 + dist.y ** 2)
        d[key]['Speed'] = dist.Dist / (dist.Time / 1e9)
        d[key]['U'] = dist.x / (dist.Time / 1e9)
        d[key]['V'] = dist.y / (dist.Time / 1e9)
        d[key] = d[key].drop('Time', axis=1)

    return d