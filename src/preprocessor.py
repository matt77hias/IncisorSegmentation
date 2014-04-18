# -*- coding: utf-8 -*-
'''
Preprocess the dental radiographs by:
    cropping to a roughly estimated region of interest, 
    reducing noise and
    enhancing contrast
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import numpy as np
import cv2
import math_utils as mu
import loader as l
import configuration as config

from matplotlib import pyplot

#cropping

def crop_by_offsets(image, top_offset, bottem_offset, left_offset, right_offset):
    '''
    Crops a given image
    @param top_offset:           the distance from the top to be cut-off.
    @param bottem_offset:        the distance from the bottem to be cut-off. 
    @param left_offset:          the distance from the left to be cut-off. 
    @param right_offset:         the distance from the right to be cut-off. 
    @return the cropped image
    '''
    [curr_height, curr_width] = image.shape[:2]
    return image[top_offset:(curr_height-bottem_offset),left_offset:(curr_width-right_offset),:]

def crop_by_offsets_and_size(image, top_offset, height, left_offset, width):
    '''
    Crops a given image
    @param top_offset:           the distance from the top to be cut-off.
    @param height:               the height of the cropped image.
    @param left_offset:          the distance from the left to be cut-off. 
    @param width:                the width of the cropped image.
    @return the cropped image
    '''
    [curr_height, curr_width] = image.shape[:2]
    return image[top_offset:(top_offset+height),left_offset:(left_offset+width),:]

def crop_by_size(image, height, width):
    '''
    Crops a given image
    @param image:                the image to be cropped
    @param width:                the crop-to-width
    @param height:               the crop-to-height
    @return the cropped image
    '''
    [curr_height, curr_width] = image.shape[:2]
    top  = (curr_height-height) /2
    left = (curr_width-width)   /2
    return crop_by_offsets_and_size(top, height, left, width)
    
def crop_by_diagonal(image, ymin, ymax, xmin, xmax):
    '''
    Crops a given image
    @param image:                the image to be cropped
    @param ymin:                 the minimal y coordinate
    @param ymax:                 the maximal y coordinate
    @param xmin:                 the minimal x coordinate
    @param xmax:                 the maximal x coordinate
    @param height:               the crop-to-height
    @return the cropped image
    '''
    [curr_height, curr_width] = image.shape[:2]
    return crop_by_offsets_and_size(image, ymin, (ymax-ymin+1), xmin, (xmax-xmin+1))
    
def learn_offsets():
    '''
    Learns the minimal y coordinate, maximal y coordinate, 
    minimal x coordinate and maximal x coordinate for the
    training samples.
    @return the learned minimal y coordinate, learned maximal y coordinate, 
            learned minimal x coordinate and learned maximal x coordinate for the
            training samples.
    '''
    xmin = ymin = float("inf")
    xmax = ymax = 0
    for j in config.get_teeth_range(): #TODO
        X = l.create_full_X(nr_tooth=j)
        for i in range(X.shape[0]):
            xCoords, yCoords = mu.extract_coordinates(X[i,:])
            #looping twice with one check has approximately same
            #complexity as looping once with two checks
            xmin = min(xmin, np.amin(xCoords))
            xmax = max(xmax, np.amax(xCoords))
            #looping twice with one check has approximately same
            #complexity as looping once with two checks
            ymin = min(ymin, np.amin(yCoords))
            ymax = max(ymax, np.amax(yCoords))
    return (ymin, ymax, xmin, xmax)

top_safety_offset = 20
bottem_safety_offset = 20
left_saftey_offset = 20
right_safety_offset = 20  
      
def learn_offsets_safe():
    '''
    Learns the minimal y coordinate, maximal y coordinate, 
    minimal x coordinate and maximal x coordinate for the
    training samples but add some safety margins.
    @return the learned minimal y coordinate, learned maximal y coordinate, 
            learned minimal x coordinate and learned maximal x coordinate for the
            training samples with added safety margins.
    '''
    (ymin, ymax, xmin, xmax) = learn_offsets()
    return ((ymin-top_safety_offset), (ymax+bottem_safety_offset), (xmin-left_saftey_offset), (xmax+right_safety_offset))

#reducing noise

def reduce_noise():
    '''
    TODO
    '''

#stretching contrast

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
    (ymin, ymax, xmin, xmax) = learn_offsets_safe()

    image_path = config.get_fname_radiograph(6)
    image = cv2.imread(image_path)
    cropped_image = crop_by_diagonal(image, ymin, ymax, xmin, xmax)
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    
    resulting_image = stretch_contrast(cropped_image)
    cv2.imshow("Stretched Contrast Image", resulting_image)
    cv2.waitKey(0)