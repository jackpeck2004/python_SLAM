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
        self.frame = frame
        gaussian_images, dog_images = self.compute_scale_space_and_image_pyramid()
        self.plot_images(dog_images[0])

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
    def compute_scale_space_and_image_pyramid(self):
        # 1. Apply the gaussian blur of sigma to obtain the base image
        base_frame = cv2.GaussianBlur(self.frame, (0, 0), sigmaX=self.sigma, sigmaY=self.sigma)

        # 2. Obtain number of octaves (how many times can the image be halved to have at least a 1 px image size)
        shortest_side = min(self.frame.shape)
        num_octaves = math.floor(math.log(shortest_side, 2)) - 1

        # 3. Generate the Gaussian kernels (how much each image in the octave has to be blurred)
        images_per_octave = self.intervals + 3 # account for the previous and next image in the same layer 
        # k = math.sqrt(2) # TODO: possibly change it to 2 ** (1/intervals)
        k = 2 ** (1. / self.intervals)
        g_kernels = np.zeros(images_per_octave) # create a empty array which will contain the kernels for the current layer of the pyramid
        g_kernels[0] = self.sigma # set the kernel for the first image to sigma
        
        # fill in the kernels with the values needded from the previous image to reach k^n * sigma
        for i in range(1, images_per_octave):
            sigma_prev = (k ** (i - 1)) * self.sigma
            sigma_to_obtain = (k ** i) * self.sigma
            kernel = sigma_to_obtain ** 2 - sigma_prev ** 2
            g_kernels[i] = kernel

        print(g_kernels)

        # 4. Create the Gaussian Images (blured)
        g_images = [] # create the array which will be filled with gaussian blured images

        # for each octave
        image = base_frame
        for octave in range(1, num_octaves):
            print("octave", octave)
            # append the first image
            g_images_octave = [image]

            # for each gaussian_kernel, iterativley additionally blur the previous image starting from the base one
            for kernel in g_kernels:
                print("kernel", kernel)
                image = cv2.GaussianBlur(image, (0,0), sigmaX=kernel, sigmaY=kernel) # create the blured image and overwrite the current one
                g_images_octave.append(image)

            g_images.append(g_images_octave)
            
            octave_base = np.array(g_images_octave[-3]) # set the octave base as the third to last layer
            h, w = octave_base.shape
            height = h // 2
            width = w // 2
            image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

        g_images = np.array(g_images, dtype=object)

        # 5. Create the DoG(Difference of Gaussian) images (subtract consecutive images from previous step)
        dog_images = []

        for gaussian_images_in_octave in g_images:
            dog_images_in_octave = []
            for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                dog_images_in_octave.append(cv2.subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
                dog_images.append(dog_images_in_octave)

        dog_images = np.array(dog_images, dtype=object)

        return g_images, dog_images

    def plot_images(self, images):
        fig = plt.figure(figsize=(20, 15))
        rows, cols = 4, 3
        for idx, image in enumerate(images):
            fig.add_subplot(rows, cols, idx + 1)
            plt.imshow(image, cmap="gray")
            plt.axis('off')
            plt.title("Image " + str(idx))
        plt.show()
