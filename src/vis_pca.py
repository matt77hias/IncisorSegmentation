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
import configuration as c
import preprocessor as pre
import procrustes_analysis as pa
import principal_component_analysis as pca

from matplotlib import pyplot

def store_plotted_vary_mode_param(closed_curve=False):
    '''
    Stores the plots of the effects of varying one mode parameter / shape parameter
    in -3 s.d. / -2 s.d. / -1 s.d. / M / +1 s.d. / +2 s.d. / +3 s.d.
    in the model coordinate frame for all the teeth.
    @param closed_curve:        must the curve be closed
    '''
    for t in c.get_teeth_range():
        M, Y = pa.PA(l.create_full_X(nr_tooth=t))
        E, W, MU = pca.pca_percentage(Y)
        
        YS = np.zeros(np.array([E.shape[0], 7, c.get_nb_dim()]))
        for e in range(YS.shape[0]):
            se = math.sqrt(E[e])
            C = np.zeros(W.shape[1])
            C[e] = 1
            j = 0
            for i in range(-3, 4):
                YS[e,j,:] = pca.reconstruct(W, C*i*se, M)
                j += 1   
        (ymin, ymax, xmin, xmax) = pre.learn_offsets(YS)
        ymin -= 0.01
        ymax += 0.01
        xmin -= 0.01
        xmax += 0.01
    
        for e in range(YS.shape[0]):
            pyplot.figure()
            for j in range(1,YS.shape[1]+1):
                xCoords, yCoords = mu.extract_coordinates(YS[e,(j-1),:])
                if (closed_curve):
                    xCoords = mu.make_circular(xCoords)
                    yCoords = mu.make_circular(yCoords)
                # x coordinates , y coordinates
                pyplot.subplot(1, 7, j)
                pyplot.plot(xCoords, yCoords, '-+r')
                pyplot.axis([xmin, xmax, ymin, ymax])
                pyplot.gca().set_aspect('equal', adjustable='box')
                if j == 1: 
                    pyplot.ylabel('y\'')
                if j == 4:
                    pyplot.title('Tooth nr: ' + str(t) + ' | ' + 'Eigenvalue: ' + str(E[e]))
                    pyplot.xlabel('x\'')
                frame = pyplot.gca()
                if j%2 == 0: 
                    frame.axes.get_xaxis().set_ticks([])
                else: 
                    frame.axes.get_xaxis().set_ticks([xmin, xmax])
                pyplot.gca().invert_yaxis()
                #pyplot.subplots_adjust(right=1)
            fname = c.get_fname_vis_pca(t, (e+1))
            pyplot.savefig(fname, bbox_inches='tight')
            pyplot.close()

def plot_vary_mode_param(nr_tooth=1, closed_curve=False):
    '''
    Plots the effects of varying one mode parameter / shape parameter
    in -3 s.d. / -2 s.d. / -1 s.d. / M / +1 s.d. / +2 s.d. / +3 s.d.
    in the model coordinate frame for the given tooth.
    @param nr_tooth:            the number ot the tooth
    @param closed_curve:        must the curve be closed
    '''
    M, Y = pa.PA(l.create_full_X(nr_tooth=nr_tooth))
    E, W, MU = pca.pca_percentage(Y)
    
    YS = np.zeros(np.array([E.shape[0], 7, c.get_nb_dim()]))
    for e in range(YS.shape[0]):
        se = math.sqrt(E[e])
        C = np.zeros(W.shape[1])
        C[e] = 1
        j = 0
        for i in range(-3, 4):
            YS[e,j,:] = pca.reconstruct(W, C*i*se, M)
            j += 1   
    (ymin, ymax, xmin, xmax) = pre.learn_offsets(YS)
    ymin -= 0.01
    ymax += 0.01
    xmin -= 0.01
    xmax += 0.01

    for e in range(YS.shape[0]):
        pyplot.figure()
        for j in range(1,YS.shape[1]+1):
            xCoords, yCoords = mu.extract_coordinates(YS[e,(j-1),:])
            if (closed_curve):
                xCoords = mu.make_circular(xCoords)
                yCoords = mu.make_circular(yCoords)
            # x coordinates , y coordinates
            pyplot.subplot(1, 7, j)
            pyplot.plot(xCoords, yCoords, '-+r')
            pyplot.axis([xmin, xmax, ymin, ymax])
            pyplot.gca().set_aspect('equal', adjustable='box')
            if j == 1: 
                pyplot.ylabel('y\'')
            if j == 4:
                pyplot.title('Tooth nr: ' + str(nr_tooth) + ' | ' + 'Eigenvalue: ' + str(E[e]))
                pyplot.xlabel('x\'')
            frame = pyplot.gca()
            if j%2 == 0: 
                frame.axes.get_xaxis().set_ticks([])
            else: 
                frame.axes.get_xaxis().set_ticks([xmin, xmax])
            pyplot.gca().invert_yaxis()
            #pyplot.subplots_adjust(right=1)
        pyplot.show()

if __name__ == '__main__':
    store_plotted_vary_mode_param(closed_curve=True)
