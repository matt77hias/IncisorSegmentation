'''
Contains some visualization functions for displaying the results
of the Procrustes Analysis
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import numpy as np
import pylab
import math_utils as mu
import loader as l
import procrustes_analysis as pa
            
def plot_mean(M, closed_curve=False):
    '''
    Plots the landmarks corresponding to the mean shape
    in the model coordinate frame.
    @param M:                   the mean shape in the model coordinate frame
    @param closed_curve:        must the curve be closed
    '''
    xCoords, yCoords = mu.extract_coordinates(M)
    if (closed_curve):
        xCoords = mu.make_circular(xCoords)
        yCoords = mu.make_circular(yCoords)
    
    pylab.figure(2)
    # x coordinates , y coordinates
    pylab.plot(xCoords, yCoords, '-+r')
    pylab.gca().invert_yaxis()
    pylab.axis('equal')
    pylab.show()
    
def plot_all(M, X, closed_curve=False):
    '''
    Plots the landmarks corresponding to the mean shape
    together with all the training samples in the model
    coordinate frame
    @param M:                   the mean shape in the model coordinate frame
    @param X:                   the training samples in the model coordinate frame
    @param closed_curve:        must the curve be closed
    '''
    xCoords, yCoords = mu.extract_coordinates(M)
    if (closed_curve):
        xCoords = mu.make_circular(xCoords)
        yCoords = mu.make_circular(yCoords)
    
    pylab.figure(2)
    # x coordinates , y coordinates
    pylab.plot(xCoords, yCoords, '-+r')
    for i in range(X.shape[0]):
        xCs, yCs = mu.extract_coordinates(Y[i,:])
        pylab.plot(xCs, yCs, '*g')
    
    pylab.gca().invert_yaxis()
    pylab.axis('equal')
    pylab.show()
    
if __name__ == '__main__':
    X = l.create_full_X(nr_tooth=1)
    M, Y = pa.PA(X)
    #plot_mean(M, closed_curve=True)
    plot_all(M, Y, closed_curve=True)