#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def tracks_sizes_selection(tracks, frameIdx, rgb):
    """
    Shows final frame with tracks and tracks' length histogram.
    Filters tracks based on their size
    :param tracks: drifters' tracks, 1D numpy array
    :param frameIdx: frames' indexes, 1D numpy array
    :param rgb: RGB frame
    :return: Filtered tracks and frames' indexes
    """
    # Turn tracks list into array
    tracks = np.asarray(tracks)
    frameIdx = np.asarray(frameIdx)
    # Checking and filtering trajectories
    cv2.namedWindow('Trajectories', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trajectories', 1200, 1200)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    keepOn = True
    while keepOn:
        # Compute trajectories' size
        tracksSize = []
        for tr in tracks:
            tracksSize.append(len(tr))
        tracksSize = np.asarray(tracksSize)
        # Plot last frame plus tracks
        rgbTracks = rgb.copy()
        rgbTracks = cv2.polylines(rgbTracks, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
        cv2.imshow('Trajectories', rgbTracks)
        ax.hist(tracksSize)
        ax.set_xlabel("Track's size")
        ax.set_ylabel("Occurrences")
        ax.set_title("Tracks sizes histogram")
        plt.draw()
        # Ask if satisfy with distribution
        satisfied = raw_input("Satisfied by the distribution? (yes/no): ").lower()
        if satisfied in 'yes':
            keepOn = False
        else:
            lowLimit = float(raw_input("Delete trajectories shorter than (integer): "))
            upLimit = float(raw_input("Delete trajectories greater than (integer): "))
            indexes = np.where(np.logical_or(tracksSize < lowLimit, tracksSize > upLimit))
            tracks = np.delete(tracks, indexes)
            frameIdx = np.delete(frameIdx, indexes)
            plt.cla()

    return tracks, frameIdx

def datetime_to_mattime(list_of_datetimes, debug=False):
    """Convert list of datetime64[us] to list matlab time"""
    list_of_datenum = []
    for dt in list_of_datetimes:
        mdn = dt + timedelta(days = 366)
        s = (dt.hour * (60.0*60.0)) + (dt.minute * 60.0) + dt.second
        day = 24.0*60.0*60.0
        frac = s/day
        list_of_datenum.append(mdn.toordinal() + frac)

    return list_of_datenum