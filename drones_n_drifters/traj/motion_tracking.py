#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import cv2
import numpy as np

# Parameters for motion_tracking_LK
lk_params = dict(winSize=(15, 15),  # what are those?
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#  ref. http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
#  ref. http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
feature_params = dict(maxCorners=500,  # what are those?
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
#  ref. http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
#  ref. http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack


def motion_tracking_Lucas_Kanade(tracks, frameIdx, frame_id, prev_gray, greyScaleMask,
                                 minTrackLength=100, detect_interval=5,
                                 lk_params=lk_params, feature_params=feature_params, debug=False):
    """
    Tracks motion with Lucas-Kanade sparse optical flow and returns obhects trajectories

    :param tracks: List of trajectories
    :param frameIdx: List of frame indexes
    :param frame_id: Frame's index
    :param prev_gray: Previous grey frame
    :param greyScaleMask: Grey frame
    :param minTrackLength: minimum length criteria for track's storage
    :param detect_interval: Frame interval for robust tracking
    :param lk_params: See ref. http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack
    :param feature_params: See http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack
    :return: tracks & frameIdx
    """

    if debug:
        gsm = greyScaleMask.copy()
    if len(tracks) > 0:
        img0, img1 = prev_gray, greyScaleMask
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)  # Back-tracking for match verification between frames
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        new_frameIdx = []
        for tr, ts, (x, y), good_flag in zip(tracks, frameIdx, p1.reshape(-1, 2), good):
            if not good_flag:
                # Check here and make sure long tracks do not diseappear
                if len(tr) > minTrackLength:
                    new_tracks.append(tr)
                    new_frameIdx.append(ts)
                continue
            tr.append((x, y))
            new_tracks.append(tr)
            ts.append(frame_id)
            new_frameIdx.append(ts)
            if debug:
                cv2.circle(gsm, (x, y), 2, (0, 255, 0), -1)
        tracks = new_tracks
        frameIdx = new_frameIdx
    # optical flow finds the next point which may look close to it.
    # So for a robust tracking, corner points are detected at every detect_interval frames.
    if frame_id % detect_interval == 0:
        mask = np.zeros_like(greyScaleMask)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)  # draw circle filled with black
        p = cv2.goodFeaturesToTrack(greyScaleMask, mask=mask, **feature_params)
        if p is not None:  # if new tracks it adds it
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])
                frameIdx.append([frame_id])

    if debug:
        cv2.namedWindow('motion tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('motion tracking', 1200, 1200)
        cv2.polylines(gsm, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
        x = 20
        y = 20
        s = 'track count: %d' % len(tracks)
        cv2.putText(gsm, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(gsm, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.imshow('motion tracking', gsm)

    return tracks, frameIdx