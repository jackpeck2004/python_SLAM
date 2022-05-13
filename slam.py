#!/usr/bin/env python3
import cv2
import sys
from sift import Sift

cap = cv2.VideoCapture("video.mp4")
sigma = 1.6
intervals = 3
s = Sift(sigma, intervals)

# If there is a problem with the capture, exit
if (cap.isOpened() == False):
    sys.stderr.write("Error opening video or file stream")
    exit(1)

# draw the keypoint circle to the frame
def draw_keypoint_to_frame(frame, pos):
    cv2.circle(frame, pos, 5, (0, 255, 0))

success, frame = cap.read()

gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# TODO: extract keypoints
#kp, des = sift(gray_frame)
s.sift(gray_frame)
# draw the keypoints

'''
while (True): # show the frame
    if(success):
        cv2.imshow("SLAM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

while True:
    success,frame=cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # TODO: extract keypoints
    kp, des = sift(gray_frame)
    # draw the keypoints

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
'''
