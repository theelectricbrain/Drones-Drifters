#!/usr/bin/python2.7
# encoding: utf-8

from __future__ import division

import cv2
import skvideo.io

import numpy as np

#cap = cv2.VideoCapture("/home/grumpynounours/Desktop/Electric_Brain/measurements/pumkin_passing_cut.avi")
# quick fix
cap = skvideo.io.VideoCapture("/home/grumpynounours/Desktop/Electric_Brain/measurements/pumkin_passing_test.MOV")

# Define the codec and create VideoWriter object
fps = round(cap.info['streams'][0]['nb_frames']/float(cap.info['format']['duration']))
#out = skvideo.io.VideoWriter('/home/grumpynounours/Desktop/output.MOV', fps=fps, frameSize=(cap.width, cap.height))
out = cv2.VideoWriter('/home/grumpynounours/Desktop/video.avi',-1,fps,(cap.width,cap.height))

#color detection
# RBG range value for Orange pumkin
colorBounds = ([180, 69, 0], [255, 215, 255]) # RGB
whiteBounds = ([245, 245, 245], [255, 255, 255])
# Dilation kernel
dilationKernel = np.ones((501, 501))


# create NumPy arrays from the boundaries
lower = np.array(colorBounds[0], dtype="uint8")
upper = np.array(colorBounds[1], dtype="uint8")
lowerW = np.array(whiteBounds[0], dtype="uint8")
upperW = np.array(whiteBounds[1], dtype="uint8")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        #Color detection, here pumkin orange
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        maskPumkins = cv2.inRange(rgb, lower, upper)
        #Mask-out white
        # Finding white patches
        grays = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        maskWhite = cv2.inRange(rgb, lowerW, upperW)
        #  Dilate white patches
        dilatedMaskWhite = cv2.bitwise_not(cv2.dilate(maskWhite, dilationKernel))
        #Resulting mask
        mask = cv2.bitwise_and(maskPumkins, maskPumkins, mask = dilatedMaskWhite)
        ###test###
        # Plotting
        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("test", mask)
        cv2.waitKey(1)
        ###test###
        #Write to video
        print "save video..."
        out.write(mask)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()



        #Motion capture
        if counter == 0:
            prvs = mask
            hsv = np.zeros_like(mask)
            hsv[..., 1] = 255
            counter += 1
        else:
            next = mask
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            prvs = next
            ###test###
            # Plotting
            cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("test", bgr)
            cv2.waitKey(1)
            ###test###



        # Plotting
        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
        #cv2.imshow("image", rgb)
        cv2.imshow("test", mask)
        #cv2.resizeWindow('mask', 1200,1200)
        cv2.waitKey(1)

# find the colors within the specified boundaries

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # convert in RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # the mask
        mask = cv2.inRange(rgb, lower, upper)
        output = cv2.bitwise_and(rgb, rgb, mask=mask)
        # show the images
        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("test", np.hstack([rgb, output]))
        cv2.waitKey(1)



# motion capture
counter = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        if counter == 0:
            prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            counter += 1
        else:
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            cv2.imshow('frame', bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png', frame)
                cv2.imwrite('opticalhsv.png', bgr)
            prvs = next

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()