# -*- coding: utf-8 -*-
'''
Contains some visualization functions for getting used
with the given data.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import cv2
import numpy as np
import pylab

import loader as l
import configuration as c

def extract_landmarks(X, nr_trainingSample=1):
    nb = c.get_nb_landmarks()
    xCoords = np.zeros(nb)
    yCoords = np.zeros(nb)
    nr_trainingSample -= 1
    for i in range(nb):
        xCoords[i] = X[nr_trainingSample, (2*i)]  
        yCoords[i] = X[nr_trainingSample, (2*i+1)]
    return (xCoords, yCoords)
                  
def plot_landmarks(X, nr_trainingSample=1):
    xCoords, yCoords = extract_landmarks(X, nr_trainingSample)
    # x coordinates , y coordinates
    pylab.plot(xCoords, yCoords, '-r')
    pylab.gca().invert_yaxis()
    pylab.axis('equal')
    pylab.show()
    
def display_landmarks(X, nr_trainingSample=1, color=np.array([0, 0, 255])):
    s = "/"
    if (nr_trainingSample < 10):
        s = "/0"
    fname = c.get_dir_radiographs() + s + str(nr_trainingSample) + '.tif'
    img = cv2.imread(fname)
    
    xCoords, yCoords = extract_landmarks(X, nr_trainingSample)
    for i in range(c.get_nb_landmarks()):
        #y coordinate , x coordinate
        img[yCoords[i], xCoords[i], :] = color
    #cv2.imshow("test", img)
    cv2.imwrite("test.tif", img)
      
if __name__ == '__main__':
    #test
    X = l.create_full_X()
    plot_landmarks(X)
    display_landmarks(X)