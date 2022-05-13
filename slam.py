#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

cap = cv2.VideoCapture("video.mp4")
sigma = 10
intervals = 3

# If there is a problem with the capture, exit
if (cap.isOpened() == False):
    sys.stderr.write("Error opening video or file stream")
    exit(1)

# draw the keypoint circle to the frame
def draw_keypoint_to_frame(frame, pos):
    cv2.circle(frame, pos, 5, (0, 255, 0))

def plot_images(images):
    fig = plt.figure(figsize=(30, 20))
    rows, cols = 4, 3
    for idx, image in enumerate(images):
        fig.add_subplot(rows, cols, idx + 1)
        plt.imshow(image, cmap="gray")
        plt.axis('off')
        plt.title("Image " + str(idx))
    plt.show()

'''
Implementation of the sift algorithm for feature detection
Values:
sigma = 1.6
k = sqrt(2)
Steps:
1. Apply the gaussian blur of sigma to obtain the base image
2. Obtain number of octaves
3. Generate the Gaussian kernels (how much each image in the octave has to be blurred)
4. Create the Gaussian Images (blured)
5. Create the DoG(Difference of Gaussian) images (subtract consecutive images from previous step)
'''
def sift(frame, sigma, intervals):
    # 1. Apply the gaussian blur of sigma to obtain the base image
    base_frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 2. Obtain number of octaves (how many times can the image be halved to have at least a 1 px image size)
    shortest_side = min(frame.shape)
    num_octaves = math.floor(math.log(shortest_side, 2)) - 1

    # 3. Generate the Gaussian kernels (how much each image in the octave has to be blurred)
    images_per_octave = intervals + 3 # account for the previous and next image in the same layer 
    k = math.sqrt(2) # TODO: possibly change it to 2 ** (1/intervals)
    # k = 2 ** (1. / intervals)
    g_kernels = [0] * images_per_octave # create a empty array which will contain the kernels for the current layer of the pyramid
    g_kernels[0] = sigma # set the kernel for the first image to sigma
    
    # fill in the kernels with the values needded from the previous image to reach k^n * sigma
    for i in range(1, images_per_octave):
        sigma_prev = (k ** (i - 1)) * sigma
        sigma_to_obtain = (k ** i) * sigma
        kernel = sigma_to_obtain ** 2 - sigma_prev ** 2
        g_kernels[i] = kernel

    # 4. Create the Gaussian Images (blured)

    # 5. Create the DoG(Difference of Gaussian) images (subtract consecutive images from previous step)

    plot_images([frame, base_frame])

success, frame = cap.read()

gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# TODO: extract keypoints
#kp, des = sift(gray_frame)
sift(gray_frame, sigma, intervals)
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
