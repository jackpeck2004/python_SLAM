import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

class Sift():
    def __init__(self, sigma, intervals):
        self.sigma = sigma
        self.intervals = intervals

    '''
    Implementation of the sift algorithm for feature detection
    '''
    def sift(self, frame):
        # 1. Apply the gaussian blur of sigma to obtain the base image
        base_frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)

        # 2. Obtain number of octaves (how many times can the image be halved to have at least a 1 px image size)
        shortest_side = min(frame.shape)
        num_octaves = math.floor(math.log(shortest_side, 2)) - 1

        # 3. Generate the Gaussian kernels (how much each image in the octave has to be blurred)
        images_per_octave = self.intervals + 3 # account for the previous and next image in the same layer 
        k = math.sqrt(2) # TODO: possibly change it to 2 ** (1/intervals)
        # k = 2 ** (1. / intervals)
        g_kernels = [0] * images_per_octave # create a empty array which will contain the kernels for the current layer of the pyramid
        g_kernels[0] = self.sigma # set the kernel for the first image to sigma
        
        # fill in the kernels with the values needded from the previous image to reach k^n * sigma
        for i in range(1, images_per_octave):
            sigma_prev = (k ** (i - 1)) * self.sigma
            sigma_to_obtain = (k ** i) * self.sigma
            kernel = sigma_to_obtain ** 2 - sigma_prev ** 2
            g_kernels[i] = kernel

        # 4. Create the Gaussian Images (blured)

        # 5. Create the DoG(Difference of Gaussian) images (subtract consecutive images from previous step)

        self.plot_images([frame, base_frame])

    '''
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
    def scale_space_and_image_pyramid():
        pass

    def plot_images(self, images):
        fig = plt.figure(figsize=(30, 20))
        rows, cols = 4, 3
        for idx, image in enumerate(images):
            fig.add_subplot(rows, cols, idx + 1)
            plt.imshow(image, cmap="gray")
            plt.axis('off')
            plt.title("Image " + str(idx))
        plt.show()
