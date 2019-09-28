# starimage.py implements the StarImage class, which does things to a star track image that facilitate
# finding the north star location 
# 6 September 2019

# imports
import numpy as np
import cv2
import scipy.signal as signal
import seaborn as sns
from scipy.optimize import minimize
from numba import cuda


# Image class, where most of the functionality occurs
class StarImage:

    # configure to environment
    device_available = True 

    # define Sobel edge detection kernels
    x_edge_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) # horizontal edges
    y_edge_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) # vertical edges
    #if device_available: # copy sobel edge kernels to device
    #    d_x_edge_kernel = cuda.to_device(x_edge_kernel) 
    #    d_y_edge_kernel = cuda.to_device(y_edge_kernel)
 
    def __init__(self, path, filename):
        self.gray_img = cv2.imread(path + filename, cv2.IMREAD_GRAYSCALE) # get grayscale img
        self.height, self.width = self.gray_img.shape
        self.path = path
        self.filename = filename
        if self.device_available: # copy sobel edge kernels and gray_img to device
            self.d_gray_img = cuda.to_device(self.gray_img)

    # This fn makes the x and y edge detected versions of the image
    def convolve(self):
        self.x_edge_img = signal.convolve2d(self.gray_img, self.x_edge_kernel, mode='same')
        self.y_edge_img = signal.convolve2d(self.gray_img, self.y_edge_kernel, mode='same') 

    # This fn removes edges facing up, so that each star track only gives one edge. Optional to use
    def removeUpwardEdges(self):
        for j in range(self.height):
            for i in range(self.width):
                if self.y_edge_img[j, i] < 0:
                    self.x_edge_img[j, i] = 0
                    self.y_edge_img[j, i] = 0

    # This fn makes the polar coords of the edge vectors of the image
    def makePolar(self):
        self.magnitude_img = np.sqrt(self.x_edge_img**2 + self.y_edge_img**2)
        self.orientation_img = np.arctan2(self.x_edge_img, self.y_edge_img) + np.pi

    # jThis fn makes the polar coords of the edge vectors of the image
    @cuda.jit(device=True)
    def cuda_makeMagnitude(self):
        # copy edge images over. Should be cleaned up once they get made on GPU
        self.d_x_edge_img = cuda.to_device(self.x_edge_img.flatten())
        self.d_y_edge_img = cuda.to_device(self.y_edge_img.flatten())

        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        self.d_magnitude_img = cuda.device_array(self.height * self.width)
        for i in range(start, self.d_gray_img.shape[0], stride):
            self.d_magnitude_img[i] = math.sqrt(self.d_x_edge_img[i]**2 + self.d_y_edge_img[i]**2)

        # reshape and send array back to CPU memory. Clean up later
        arr = self.d_magnitude_image.reshape((self.height, self.width))
        arr.copy_to_host(self.magnitude_img_from_gpu)

    # This fn makes an array in the shape of gray_img where the values match the indices, for use in the cost fn 
    # This could be accomplished by using the same for loop structure in the cost fn,
    # but its faster to do it once then use numpy in the cost fn
    def makeArrayLocationMatrices(self):
        self.x_loc = np.zeros_like(self.gray_img, dtype='float64')
        self.y_loc = np.zeros_like(self.gray_img, dtype='float64')
        for j in range(self.height):
            for i in range(self.width):
                self.x_loc[j, i] = i
                self.y_loc[j, i] = j

    # This fn takes a coordinate in the form of pixel indices [x, y] and returns the cost fn value for that pixel
    # The pixel indices represent the guess for the location of the north star and need not be within the image
    def cost(self, coord):

        # init vector that points from its location to coord
        W_x = np.zeros_like(self.gray_img, dtype='float64') # init array to zeros
        W_y = np.zeros_like(self.gray_img, dtype='float64') # init array to zeros

        # W[x, y] is a unit vector that points from [x, y] to coord.
        # It is created by dividing the vector, coord - [x, y], by magnitude, except where magnitude == 0  
        np.true_divide(coord[0] - self.x_loc, self.magnitude_img, out=W_x, where=self.magnitude_img!=0)
        np.true_divide(coord[1] - self.y_loc, self.magnitude_img, out=W_y, where=self.magnitude_img!=0)

        # Evaluate cost fn at coord
        # This is a modified dot product of W and the edge vector
        # Modified because we take the absolute value since direction is +pi ambiguous
        cost_array = abs(W_x * self.x_edge_img + W_y * self.y_edge_img)
        return np.sum(cost_array)

    # This fn uses a minimization method to find the coordinates of the north star
    def findNorthStar(self, guess=None):
        if None == guess:
            guess = (self.width/2, self.height/2) # Guess the middle of the image
        min_result = minimize(self.cost, guess, method='nelder-mead')
        self.north_star_loc = tuple([int(x) for x in tuple(min_result.x)])


    # ----------------------- VISUALIZATION METHODS ---------------------------------------------------------

    # This fn creates a copy of the original image with the predicted location of the north star marked
    # with a red circle
    def makeOutputImage(self):
        filename = self.filename.split('.')
        out_filename = filename[0] + '_w_north_star.' + filename[1] 
        color_img = cv2.imread(self.path + self.filename, cv2.IMREAD_COLOR) # get color image
        cv2.circle(color_img, self.north_star_loc, 4, (0, 0, 255)) # add circle
        cv2.imwrite(self.path + out_filename, color_img) # write image

    # This fn creates and saves an image where the magnitude is mapped to val and the orientation is mapped to
    # hue for a HSV image
    def makeOrientedEdgeImage(self, out_filename):

        # define hue, saturation, value arrays in the shape of the image
        hue = self.orientation_img * 179/(2 * np.pi) # This is the hue, mapped from orientation
        sat = np.zeros_like(self.gray_img) + 255 # Set to 255
        val = self.magnitude_img * 255/np.max(self.magnitude_img) # scale magnitude to 0-255

        hsv_img = np.zeros((self.height, self.width, 3), dtype='float32')
        for j in range(self.height):
            for i in range(self.width):
                hsv_img[j, i, 0] = hue[j, i]
                hsv_img[j, i, 1] = sat[j, i]
                hsv_img[j, i, 2] = val[j, i]

        # write image to file
        bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        cv2.imwrite(self.path + out_filename, bgr_img)

    # This fn shows a heatmap of the cost fn over the image
    def heatmap(self):
        x = np.arange(0, self.width, 10)
        y = np.arange(0, self.height, 10)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self.cost([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        print('\nHeatmap representing cost function over image')
        sns.heatmap(Z)
