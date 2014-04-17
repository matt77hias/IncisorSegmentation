'''
Preprocess the dental radiographs by cropping to a roughly estimated region of interest, 
reducing noise and enhancing contrast
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import numpy as np
import cv2
import configuration as config
from matplotlib import pyplot

def crop(image, width, height, top_offset):
    '''
    Crops a given image to a image with a smaller width and height.
    As the incisors in the radiographs are not exactly centered, a part of the top can also be cut-off. 
    @param image:                the image to be cropped
    @param width:                the crop-to-width
    @param height:               the crop-to-height
    @param top_offset:           the distance from the top to be additionally cut-off. 
    @return the cropped image
    '''

    [curr_height, curr_width] = image.shape[:2]

    left   = np.around((curr_width - width) / 2)
    right  = left + width
    top    = np.around((curr_height - height) / 2) + top_offset
    bottom = top + height - top_offset
    
    return image[top:bottom, left:right]

def reduce_noise():
    '''
    '''

def stretch_contrast(image):
    '''
    Source: http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
    '''
    a = 0 #lower limit
    b = 255 #upper limit
    c, d = getValuesFromHistogram(image)

    factor = ((b - a) / (d - c))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_color = (image[i][j] - c) * factor + a
            if pixel_color > 255: pixel_color = 255
            if pixel_color < 0: pixel_color = 0
            image[i][j] = pixel_color
            
    return image

def getValuesFromHistogram(image):
    '''
    TODO
    Source: http://homepages.inf.ed.ac.uk/rbf/HIPR2/histgram.htm
    '''
    c = -1
    d = -1
    
    total_pixels = image.shape[0] * image.shape[1]
    lp_pixels = total_pixels * 5 / 100 #5% (5th percentile) of the pixels in the histogram will have values lower than c
    up_pixels = total_pixels * 95 / 100 #95% (95th percentile) of the pixels in the histogram will have values lower than d

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    
    pyplot.hist(image.ravel(),256,[0,256])
    pyplot.show()
    
    pixels = 0
    for i in range(hist.shape[0]):
        pixels = pixels + hist[i][0]
        if (pixels >= lp_pixels and c == -1):
            c = i
        if (pixels >= up_pixels and d == -1):
            d = i
    return c, d

if __name__ == '__main__':
    
    image_path = config.get_fname_radiograph(6)
    image = cv2.imread(image_path)
    cropped_image = crop(image, 1000, 1100, 400)
    cv2.imshow("Cropped Image", cropped_image)
    resulting_image = stretch_contrast(cropped_image)
    cv2.imshow("Stretched Contrast Image", resulting_image)
    cv2.waitKey(0)