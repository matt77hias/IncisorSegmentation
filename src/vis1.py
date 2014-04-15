# -*- coding: utf-8 -*-
'''
Contains some visualization functions for getting used
with the given data (landmarks).
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import cv2
import numpy as np
import pylab

import math_utils as mu
import loader as l
import configuration as c
                  
def plot_landmarks(X, nr_trainingSample=1, closed_curve=False):
    '''
    Plots the landmarks corresponding to the given training sample.
    @param X:                   the training samples
    @param nr_trainingSample:   the training sample to select
    @param closed_curve:        has the curve to be closed
    '''
    xCoords, yCoords = mu.extract_coordinates(X[(nr_trainingSample-1),:])
    if (closed_curve):
        xCoords = mu.make_circular(xCoords)
        yCoords = mu.make_circular(yCoords)
    
    # x coordinates , y coordinates
    pylab.plot(xCoords, yCoords, '-+r')
    pylab.gca().invert_yaxis()
    pylab.axis('equal')
    pylab.show()
    
def display_landmarks(X, nr_trainingSample=1, color=np.array([0, 0, 255])):
    '''
    Displays the landmarks corresponding to the given training sample
    on the corresponding radiograph.
    @param X:                   the training samples
    @param nr_trainingSample:   the training sample to select
    @param color:               the color used for displaying
    '''
    s = "/"
    if (nr_trainingSample < 10):
        s = "/0"
    fname = c.get_dir_radiographs() + s + str(nr_trainingSample) + '.tif'
    img = cv2.imread(fname)
    
    xCoords, yCoords = mu.extract_coordinates(X[(nr_trainingSample-1),:])
    for i in range(c.get_nb_landmarks()):
        #y coordinate , x coordinate
        img[yCoords[i], xCoords[i], :] = color
    #cv2.imshow("test", img)
    #writing instead of showing, because the displayed image is too large
    cv2.imwrite("test.tif", img)
      
if __name__ == '__main__':
    #test
    X = l.create_full_X()
    plot_landmarks(X, closed_curve=True)
    display_landmarks(X)