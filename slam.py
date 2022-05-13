#!/usr/bin/env python3
import cv2
import numpy as np
import sys

cap = cv2.VideoCapture("video.mp4")

# If there is a problem with the capture, exit
if (cap.isOpened() == False):
    sys.stderr.write("Error opening video or file stream")
    exit(1)

# draw the keypoint circle to the frame
def draw_keypoint_to_frame(frame, pos):
    cv2.circle(frame, pos, 5, (0, 255, 0))

'''
Implementation of the sift algorithm for feature detection
'''
def sift(frame):
    sift = cv2.SIFT_create()
    # Generate scale-space 
    kp, des = sift.detectAndCompute(frame, None)
    return kp, des

while True:
    success,frame=cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # TODO: extract keypoints
    kp, des = sift(gray_frame)
    # draw the keypoints
    # draw_keypoint_to_frame(frame, (1080//2, 1920//2))
    '''
    for p in kp:
        u, v = p.pt
        print(u, v)
        draw_keypoint_to_frame(frame, (u, v))
    '''

    cv2.drawKeypoints(gray_frame, kp, frame)

    # show the frame
    if(success):
        cv2.imshow("SLAM", frame)

    # wait for exit
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

# cleanup
cap.release()        
cv2.destroyAllWindows()
