'''
Contains some visualization functions for displaying the results
of the Procrustes analysis
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
    Plots the landmarks corresponding to the given training sample.
    @param X:                   the training samples
    @param nr_trainingSample:   the training sample to select
    @param closed_curve:        has the curve to be closed
    '''
    xCoords, yCoords = mu.extract_coordinates(M)
    if (closed_curve):
        xCoords = mu.make_circular(xCoords)
        yCoords = mu.make_circular(yCoords)
    
    # x coordinates , y coordinates
    pylab.plot(xCoords, yCoords, '-+r')
    pylab.gca().invert_yaxis()
    pylab.axis('equal')
    pylab.show()
    
def plot_all(M, X, closed_curve=False):
    '''
    Plots the landmarks corresponding to the given training sample.
    @param X:                   the training samples
    @param nr_trainingSample:   the training sample to select
    @param closed_curve:        has the curve to be closed
    '''
    xCoords, yCoords = mu.extract_coordinates(M)
    if (closed_curve):
        xCoords = mu.make_circular(xCoords)
        yCoords = mu.make_circular(yCoords)
    
    # x coordinates , y coordinates
    pylab.plot(xCoords, yCoords, '-+r')
    for i in range(X.shape[0]):
        xCs, yCs = mu.extract_coordinates(mu.align_with(mu.centre_onOrigin(X[i,:]), M))
        pylab.plot(xCs, yCs, '*b')
    
    pylab.gca().invert_yaxis()
    pylab.axis('equal')
    pylab.show()
    
if __name__ == '__main__':
    X = l.create_full_X()
    #plot_mean(pa.PA(X), closed_curve=True)
    #plot_all(pa.PA(X), X, closed_curve=True)