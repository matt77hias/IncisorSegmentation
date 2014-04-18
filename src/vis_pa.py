'''
Contains some visualization functions for displaying the results
of the Procrustes Analysis
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import math_utils as mu
import loader as l
import procrustes_analysis as pa

from matplotlib import pyplot
            
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
    
    pyplot.figure(2)
    # x coordinates , y coordinates
    pyplot.plot(xCoords, yCoords, '-+r')
    pyplot.gca().invert_yaxis()
    pyplot.axis('equal')
    pyplot.show()
    
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
    
    pyplot.figure(2)
    # x coordinates , y coordinates
    pyplot.plot(xCoords, yCoords, '-+r')
    for i in range(X.shape[0]):
        xCs, yCs = mu.extract_coordinates(Y[i,:])
        pyplot.plot(xCs, yCs, '*g')
    
    pyplot.gca().invert_yaxis()
    pyplot.axis('equal')
    pyplot.show()
    
if __name__ == '__main__':
    X = l.create_full_X(nr_tooth=1)
    M, Y = pa.PA(X)
    #plot_mean(M, closed_curve=True)
    plot_all(M, Y, closed_curve=True)