'''
Contains some visualization functions for displaying the results
of the Principal Component Analysis
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import numpy as np
import math
import math_utils as mu
import loader as l
import procrustes_analysis as pa
import principal_component_analysis as pca

from matplotlib import pyplot

def plot_vary_mode_param(M, W, C, e, closed_curve=False):
    '''
    Plots the effects of varying one mode parameter / shape parameter
    in -3 s.d. / -2 s.d. / -1 s.d. / M / +1 s.d. / +2 s.d. / +3 s.d.
    in the model coordinate frame
    @param M:                   the mean shape in the model coordinate frame
    @param W:                   the eigenvectors in the model coordinate frame
    @param C                    a coefficient vector with all entries equal to
                                zero, except one entry equal to one (at the position
                                of the given eigenvalue)
    @param e                    an eigenvalue 
    @param closed_curve:        must the curve be closed
    '''
    se = math.sqrt(e)
    
    pyplot.figure(3)
    
    j = 1
    for i in range(-3, 4):
        Y = pca.reconstruct(W, C*i*se, M)
        xCoords, yCoords = mu.extract_coordinates(Y)
        if (closed_curve):
            xCoords = mu.make_circular(xCoords)
            yCoords = mu.make_circular(yCoords)
        # x coordinates , y coordinates
        pyplot.subplot(1, 7, j)
        pyplot.plot(xCoords, yCoords, '-+r')     
        pyplot.gca().invert_yaxis()
        pyplot.axis('equal')
        j += 1

    pyplot.show()

if __name__ == '__main__':
    X = l.create_full_X(nr_tooth=1)
    M, Y = pa.PA(X)
    E, W, MU = pca.pca_percentage(Y)
    
    #Plot
    i = 0
    C = np.zeros(W.shape[1])
    C[i] = 1
    plot_vary_mode_param(M, W, C, E[i], closed_curve=True)
