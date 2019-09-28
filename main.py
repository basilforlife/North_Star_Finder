# This is a clean rewrite of find_north_star.py, with the goal of CUDA-izing the algorithm later
# 7 September 2019


# Imports
from starimage import StarImage # custom class
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2
import seaborn as sns
import numpy as np
from numpy import testing


# Params
PATH = '/home/roy/North_Star_Finder/test_images/'
FILENAME = 'test_1.png'
OUT_FILENAME = 'test_1_w_north_star.png'


# Operations

# find north star
img = StarImage(PATH, FILENAME)
img.convolve()
img.makePolar()
img.makeArrayLocationMatrices()
img.findNorthStar()
print('north_star_loc = ' + repr(img.north_star_loc))

# test to see if magnitude_img calc on GPU is correct
#img.cuda_makeMagnitude()
#testing.assert_almost_equal(img.magnitude_img, self.magnitude_img_from_gpu)

# save image with north star location indicated
img.makeOutputImage()


