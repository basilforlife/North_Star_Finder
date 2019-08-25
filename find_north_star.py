import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import seaborn as sns


#cv2.namedWindow('STARS', cv2.WINDOW_NORMAL)
#cv2.imshow('STARS',grayscale_img)
#cv2.waitKey(0)
#cv2.destroyWindow('STARS')

# Parameters
IMG_NAME = 'star_trail.jpg'

# import image
grayscale_img = cv2.imread(IMG_NAME,cv2.IMREAD_GRAYSCALE) # get grayscale img
img = cv2.imread(IMG_NAME,cv2.IMREAD_COLOR) # get color image
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # get hsv image

# get img dimensions
height, width = grayscale_img.shape

#S = 0.3
#grayscale_img_s = cv2.resize(grayscale_img,(int(S*width), int(S*height)), interpolation = cv2.INTER_AREA)
#hsv_img_s = cv2.resize(hsv_img,(int(S*width), int(S*height)), interpolation = cv2.INTER_AREA)
#
#width = int(S*width)
#height = int(S*height)


width = 750
height = 320
grayscale_img_s = grayscale_img[0:height,0:width]
hsv_img_s = hsv_img[0:height,0:width]
b_hsv_img = np.copy(hsv_img_s)
hsv_img_s_sol = np.copy(hsv_img_s)



img = img[0:height,0:width]

print('\n\nOriginal Image')
print('--Goal: Find North Star')
plt.figure(figsize=(10,15))
plt.imshow(img)
plt.show()



buffer = [0,0,0,0,0,0,0,0,0]
x_edge = np.zeros(shape=(height,width))
y_edge = np.zeros(shape=(height,width))
magnitude = np.zeros(shape=(height,width))
orientation = np.zeros(shape=(height,width))

# Loop through all pixels
for y in range(1,height-1):
    for x in range(1,width-1):
        
        # get values of 3x3 pixel square around pixel at (x,y)
#        buffer[0] = grayscale_img[y-1][x-1]
#        buffer[1] = grayscale_img[y-1][x+0]
#        buffer[2] = grayscale_img[y-1][x+1]
#        buffer[3] = grayscale_img[y+0][x-1]
#        buffer[4] = grayscale_img[y+0][x+0]
#        buffer[5] = grayscale_img[y+0][x+1]
#        buffer[6] = grayscale_img[y+1][x-1]
#        buffer[7] = grayscale_img[y+1][x+0]
#        buffer[8] = grayscale_img[y+1][x+1]
        
        
        buffer[0] = grayscale_img_s[y-1][x-1]
        buffer[1] = grayscale_img_s[y-1][x+0]
        buffer[2] = grayscale_img_s[y-1][x+1]
        buffer[3] = grayscale_img_s[y+0][x-1]
        buffer[4] = grayscale_img_s[y+0][x+0]
        buffer[5] = grayscale_img_s[y+0][x+1]
        buffer[6] = grayscale_img_s[y+1][x-1]
        buffer[7] = grayscale_img_s[y+1][x+0]
        buffer[8] = grayscale_img_s[y+1][x+1]
        
        
        
        
        # get vertical and horizontal Sobel kernel convolution
        y_edge[y][x] = buffer[0] + 2 * buffer[3] + buffer[6] \
                     - buffer[2] - 2 * buffer[5] - buffer[8]
        x_edge[y][x] = buffer[6] + 2 * buffer[7] + buffer[8] \
                     - buffer[0] - 2 * buffer[1] - buffer[2]
            
            
            
        # try this to get only one edge !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if y_edge[y][x] < 0:
            y_edge[y][x] = 0
            x_edge[y][x] = 0
        
        # get magnitude and orientation of edge
        magnitude[y][x] = np.sqrt(y_edge[y][x]**2 + x_edge[y][x]**2)
        
        # in range (0, 2pi)
        orientation[y][x] = np.arctan2(y_edge[y][x],x_edge[y][x])
        
        
        
# Create output image
max_magnitude = np.nanmax(magnitude) # find max of magnitudes, ignoring NaNs

# Loop through to create new pixels in HSV
for y in range(1,height-1):
    for x in range(1,width-1):
        val = magnitude[y][x]*255/max_magnitude
        hue = orientation[y][x]*179/(2*np.pi)
        
        # overwrite original hsv image with new values (deal with edge pixels!)!!!
#        hsv_img[y][x][0] = hue # hue
#        hsv_img[y][x][1] = 255 # saturation
#        hsv_img[y][x][2] = val # value
        
        
        
        
        hsv_img_s[y][x][0] = hue # hue
        hsv_img_s[y][x][1] = 255 # saturation
        hsv_img_s[y][x][2] = val # value
        
        
#bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

bgr_img_s = cv2.cvtColor(hsv_img_s, cv2.COLOR_HSV2BGR)

print('\n\nImage after edge detection')
print('--Color represents orientation of edge')
plt.figure(figsize=(10,15))
plt.imshow(bgr_img_s)
plt.show()




# segregate into blocks and color blocks
B_SIZE = 10
b_width = width//B_SIZE
b_height = height//B_SIZE

b_magnitude = np.zeros(shape=(b_height,b_width))
b_orientation = np.zeros(shape=(b_height,b_width))

b_img_comp  = np.zeros(shape=(b_height,b_width,2)) # x and y components

# loop through all pixels, add x and y components together in blocks
for y in range(1,b_height*B_SIZE-1):
    for x in range(1,b_width*B_SIZE-1):
        b_img_comp[y//B_SIZE][x//B_SIZE][0] = b_img_comp[y//B_SIZE][x//B_SIZE][0] + y_edge[y][x]
        b_img_comp[y//B_SIZE][x//B_SIZE][1] = b_img_comp[y//B_SIZE][x//B_SIZE][1] + x_edge[y][x]
        
# loop through blocks, find orientation and magnitude
for y in range(b_height):
    for x in range(b_width):
        
        # get magnitude and orientation of edge
        b_magnitude[y][x] = np.sqrt(b_img_comp[y][x][0]**2 + b_img_comp[y][x][1]**2)
        
        # in range (0, 2pi)
        b_orientation[y][x] = np.arctan2(b_img_comp[y][x][0],b_img_comp[y][x][1])
        
        
        
        
# find max to normalize
b_max_magnitude = np.nanmax(b_magnitude)


# Loop through pixels and color according to blocks
for y in range(b_height*B_SIZE-1):
    for x in range(b_width*B_SIZE-1):
        val = b_magnitude[y//B_SIZE][x//B_SIZE]*255/b_max_magnitude
        hue = b_orientation[y//B_SIZE][x//B_SIZE]*179/(2*np.pi)
        
        # overwrite original hsv image with new values
        b_hsv_img[y][x][0] = hue # hue
        b_hsv_img[y][x][1] = 255 # saturation
        b_hsv_img[y][x][2] = val # value
 
block_img = cv2.cvtColor(b_hsv_img, cv2.COLOR_HSV2BGR)

print('\n\nScaled down edge detected image')
print('--Scaling down the image into fewer pixels allows search for minimum to run faster')
plt.figure(figsize=(10,15))
plt.imshow(block_img)
plt.show()





# Cost() takes X, a vector representing a 2d point, and an hsv img, and returns the 
# cost (function evaluation at the point given), where the cost is a modified
# vector cross product
def Cost(X, img_components):
    
    height, width = img_components.shape[:2]
    
    W = [0,0]
    
    # X is a 2 element vector
    # the origin is at the top left of the image
    
    # loop through all blocks
    cost = 0
    for j in range(height):
        for i in range(width):
        
            # find unit vector pointing from pixel block to point
            y = X[0]-j
            x = X[1]-i
            vec_mag = np.sqrt(y**2 + x**2)
            if vec_mag == 0:
                W = [0,0]
            else:
                W[0] = y/vec_mag
                W[1] = x/vec_mag
            
            
            # find modified dot product of above vector and block vector
            # modified bc we take abs value since direction is +pi ambiguous 
            cost = cost + abs(W[0]*img_components[j][i][0]+W[1]*img_components[j][i][1])
    
    return cost


x0 = [200//B_SIZE,200//B_SIZE]
north_star = minimize(Cost, x0, args=(b_img_comp), method='nelder-mead',options={'xtol': 3, 'disp': False})


x = np.arange(0, 75, 2)
y = np.arange(0, 32, 2)
X, Y = np.meshgrid(x, y)
zs = np.array([Cost([y,x],b_img_comp) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

print('\n\nHeatmap representing cost function over image')
print('--The cost function is minimized using the Nelder-Mead method')
sns.heatmap(Z)




# Add dot to image to show location of north star
y_loc, x_loc = [int(x) for x in tuple(north_star.x*B_SIZE)]



import matplotlib.patches as patches



fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(111, aspect='equal')
ax.imshow(img)
n_star = patches.Circle((x_loc,y_loc), radius=5,edgecolor='r',linewidth=2)
ax.add_patch(n_star)
plt.show()
print('North Star Coordinates: ' + repr([int(x) for x in tuple(north_star.x*B_SIZE)]))

cv2.circle(img,(x_loc,y_loc),4,(0,0,255))
cv2.imwrite('test.png',img)
