#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division
import csv
import cv2
import skvideo.io
import numpy as np
from imgprocess.image_processing import *

# Debug flag
debug=True

# Video capture
# cap = cv2.VideoCapture("/home/grumpynounours/Desktop/Electric_Brain/measurements/pumkin_passing_cut.avi")
# quick fix
cap = skvideo.io.VideoCapture("/home/grumpynounours/Desktop/Electric_Brain/measurements/pumkin_passing_test.MOV")

# Color detection attributes
colorDetect = False  # perform color detection yes/no, true/false
#  RBG range value for Orange pumkin
colorBounds = ([180, 60, 0], [240, 220, 250])  # shades of orange
whiteBounds = ([245, 245, 245], [255, 255, 255])

# White patches masking attributes
#  Dilation kernel
dilatationKernelSize = 301

# Motion tracking attributes
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

detect_interval = 5  # Default value = 5. optical flow finds the next point which may look close to it.
                     # So for a robust tracking, corner points are detected at every detect_interval frames.
tracks = []
frame_idx = 0

# Plotting attributes
cv2.namedWindow('test',cv2.WINDOW_NORMAL)
cv2.resizeWindow('test', 1200,1200)

# # While loop color detection and surf masking
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#         #Color detection, here pumkin orange
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         maskPumkins = cv2.inRange(rgb, lower, upper)
#         #Mask-out white
#         # Finding white patches
#         grays = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
#         maskWhite = cv2.inRange(rgb, lowerW, upperW)
#         #  Dilate white patches
#         dilatedMaskWhite = cv2.bitwise_not(cv2.dilate(maskWhite, dilationKernel))
#         #Resulting mask
#         mask = cv2.bitwise_and(maskPumkins, maskPumkins, mask = dilatedMaskWhite)
#         # Plotting
#         cv2.imshow("test", mask)
#         cv2.waitKey(1)
# # Release everything if job is finished
# cap.release()
# cv2.destroyAllWindows()

# # While loop motion tracking
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         vis = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
#         if len(tracks) > 0:
#             img0, img1 = prev_gray, frame_gray
#             p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
#             p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
#             p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
#             d = abs(p0-p0r).reshape(-1, 2).max(-1)
#             good = d < 1
#             new_tracks = []
#             for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
#                 if not good_flag:
#                     continue
#                 tr.append((x, y))
#                 #if len(tr) > track_len:
#                 #    del tr[0]
#                 new_tracks.append(tr)
#                 cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
#             tracks = new_tracks
#             cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
#             #draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
#             x = 20
#             y = 20
#             s = 'track count: %d' % len(tracks)
#             cv2.putText(vis, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
#             cv2.putText(vis, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
#
#         if frame_idx % detect_interval == 0:
#             mask = np.zeros_like(frame_gray)
#             mask[:] = 255
#             for x, y in [np.int32(tr[-1]) for tr in tracks]:
#                 cv2.circle(mask, (x, y), 5, 0, -1)
#             p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
#             if p is not None:
#                 for x, y in np.float32(p).reshape(-1, 2):
#                     tracks.append([(x, y)])
#
#         frame_idx += 1
#         prev_gray = frame_gray
#         cv2.imshow('test', vis)
#
#         cv2.waitKey(1)


# While loop with both filtering and motion tracking
# While loop color detection and surf masking
#while(cap.isOpened()):
for l in range(cap.info['streams'][0]['nb_frames']):
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Color detection, here pumkin orange
        if colorDetect:
            greyScaleMask = color_detection(rgb, colorBounds=colorBounds, debug=debug)
        else:
            greyScaleMask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Mask-out white
        dilatedMaskWhite = white_patches_masking(rgb, whiteBounds=whiteBounds,
                                                      dilatationKernelSize=dilatationKernelSize,
                                                      debug=debug)
        #Resulting mask
        maskWP = cv2.bitwise_and(greyScaleMask, greyScaleMask, mask=dilatedMaskWhite)
        #Motion tracking
        frame_gray = maskWP  # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = rgb.copy()
        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                #if len(tr) > track_len:
                #    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
            #draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
            x = 20
            y = 20
            s = 'track count: %d' % len(tracks)
            cv2.putText(vis, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(vis, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

        if frame_idx % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])

        frame_idx += 1
        prev_gray = frame_gray
        cv2.imshow('test', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print "Breaking loop"
            break
# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
# Save tracks in csv
print "Saving"
myfile = open('/home/grumpynounours/Desktop/Electric_Brain/measurements/captured_tracks.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(tracks)



# #Write video...does not work!
# # Define the codec and create VideoWriter object
# fps = round(cap.info['streams'][0]['nb_frames']/float(cap.info['format']['duration']))
# #out = skvideo.io.VideoWriter('/home/grumpynounours/Desktop/output.MOV', fps=fps, frameSize=(cap.width, cap.height))
# out = cv2.VideoWriter('/home/grumpynounours/Desktop/video.avi',-1,fps,(cap.width,cap.height))
# # in while loop
# # Write to video
# print "save video..."
# out.write(mask)
# # after while loop
# out.release()
# cv2.destroyAllWindows()
#
# #Methods
# def draw_str(dst, (x, y), s):
#     cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
#     cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
#
#
#
#         #Motion capture
#         if counter == 0:
#             prvs = mask
#             hsv = np.zeros_like(mask)
#             hsv[..., 1] = 255
#             counter += 1
#         else:
#             next = mask
#             flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#             mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#             hsv[..., 0] = ang * 180 / np.pi / 2
#             hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#             bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#             prvs = next
#             ###test###
#             # Plotting
#             cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
#             cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#             cv2.imshow("test", bgr)
#             cv2.waitKey(1)
#             ###test###
#
#
#
#         # Plotting
#         cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
#         cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#         #cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
#         #cv2.imshow("image", rgb)
#         cv2.imshow("test", mask)
#         #cv2.resizeWindow('mask', 1200,1200)
#         cv2.waitKey(1)
#
# # find the colors within the specified boundaries
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#         # convert in RGB
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # the mask
#         mask = cv2.inRange(rgb, lower, upper)
#         output = cv2.bitwise_and(rgb, rgb, mask=mask)
#         # show the images
#         cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
#         cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#         cv2.imshow("test", np.hstack([rgb, output]))
#         cv2.waitKey(1)
#
#
#
# # motion capture
# counter = 0
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#         if counter == 0:
#             prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             hsv = np.zeros_like(frame)
#             hsv[..., 1] = 255
#             counter += 1
#         else:
#             next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#             mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#             hsv[..., 0] = ang * 180 / np.pi / 2
#             hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#             bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
#             cv2.imshow('frame', bgr)
#             k = cv2.waitKey(30) & 0xff
#             if k == 27:
#                 break
#             elif k == ord('s'):
#                 cv2.imwrite('opticalfb.png', frame)
#                 cv2.imwrite('opticalhsv.png', bgr)
#             prvs = next
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# cap.release()
# cv2.destroyAllWindows()