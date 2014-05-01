# -*- coding: utf-8 -*-
'''
Contains some visualization functions for getting used
with the given data (landmarks).
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import cv2
import numpy as np

import preprocessor as pre
import math_utils as mu
import loader as l
import configuration as c

from matplotlib import pyplot

def store_plotted_landmarks(closed_curve=False):
    '''
    Stores the plots of the landmarks corresponding to all the
    training samples and all the teeth in the image coordinate frame.
    The scale and limits of the axes are the same for each plot.
    @param closed_curve:        must the curve be closed
    '''
    XS = l.create_full_XS()
    (ymin, ymax, xmin, xmax) = pre.learn_offsets_safe(XS)
    
    for j in range(XS.shape[0]):
        for i in range(XS.shape[1]):
            xCoords, yCoords = mu.extract_coordinates(XS[j,i,:])
            if (closed_curve):
                xCoords = mu.make_circular(xCoords)
                yCoords = mu.make_circular(yCoords)
    
            pyplot.figure()
            pyplot.title('Training sample nr: ' + str((i+1)) + ' | ' + 'Tooth nr: ' + str((j+1)))
            # x coordinates , y coordinates
            pyplot.plot(xCoords, yCoords, '-+r')
            pyplot.axis([xmin, xmax, ymin, ymax])
            pyplot.gca().set_aspect('equal', adjustable='box')
            pyplot.xlabel('x')
            pyplot.ylabel('y')
            pyplot.gca().invert_yaxis()
            fname = c.get_fname_vis_landmark((i+1), (j+1))
            pyplot.savefig(fname, bbox_inches='tight')
            #You get a runtime warning if you open more than 20 figures
            #Closing comes with a performance penalty
            pyplot.close()
                  
def plot_landmarks(X, nr_trainingSample=1, nr_tooth=1, closed_curve=False):
    '''
    Plots the landmarks corresponding to the given training sample
    in the image coordinate frame.
    @param X:                   the training samples
    @param nr_trainingSample:   the training sample to select
    @param nr_tooth:            the number of the tooth
                                (just used for the title of the plot)
    @param closed_curve:        must the curve be closed
    '''
    xCoords, yCoords = mu.extract_coordinates(X[(nr_trainingSample-1),:])
    if (closed_curve):
        xCoords = mu.make_circular(xCoords)
        yCoords = mu.make_circular(yCoords)
    
    pyplot.figure(1)
    pyplot.title('Training sample nr: ' + str(nr_trainingSample) + ' | ' + 'Tooth nr: ' + str(nr_tooth))
    # x coordinates , y coordinates
    pyplot.plot(xCoords, yCoords, '-+r')
    pyplot.gca().invert_yaxis()
    pyplot.axis('equal')
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.show()
    
def display_landmarks(X, nr_trainingSample=1, color=np.array([0, 0, 255])):
    '''
    Displays the landmarks corresponding to the given training sample
    on the corresponding radiograph in the image coordinate frame.
    @param X:                   the training samples
    @param nr_trainingSample:   the training sample to select
    @param color:               the color used for displaying
    '''
    img = cv2.imread(c.get_fname_radiograph(nr_trainingSample))
    
    xCoords, yCoords = mu.extract_coordinates(X[(nr_trainingSample-1),:])
    for i in range(c.get_nb_landmarks()):
        #y coordinate , x coordinate
        img[yCoords[i], xCoords[i], :] = color
    #cv2.imshow("test", img)
    #writing instead of showing, because the displayed image is too large
    cv2.imwrite("test.tif", img)
      
if __name__ == '__main__':
    X = l.create_full_X(nr_tooth=1)
    #display_landmarks(X, nr_trainingSample=1)
    plot_landmarks(X, nr_trainingSample=1)
    #store_plotted_landmarks(closed_curve=True)