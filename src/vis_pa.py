'''
Contains some visualization functions for displaying the results
of the Procrustes Analysis
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import configuration as c
import loader as l
import math_utils as mu
import procrustes_analysis as pa

from matplotlib import pyplot

def store_plotted_means(closed_curve=False):
    '''
    Stores the plots of the means corresponding to all the
    the teeth in the model coordinate frame. The scale and
    limits of the axes are the same for each plot.
    @param closed_curve:        must the curve be closed
    '''
    for j in range(c.get_nb_teeth()):
        M, Y = pa.PA(l.create_full_X(nr_tooth=(j+1)))
        xCoords, yCoords = mu.extract_coordinates(M)
        if (closed_curve):
            xCoords = mu.make_circular(xCoords)
            yCoords = mu.make_circular(yCoords)
    
        pyplot.figure()
        # x coordinates , y coordinates
        pyplot.plot(xCoords, yCoords, '-+r')
        pyplot.title('Tooth nr: ' + str((j+1)))
        pyplot.xlabel('x\'')
        pyplot.ylabel('y\'')
        pyplot.gca().invert_yaxis()
        pyplot.axis('equal')
        fname = c.get_fname_vis_pa((j+1), samples_included=False)
        pyplot.savefig(fname, bbox_inches='tight')
        #You get a runtime warning if you open more than 20 figures
        #Closing comes with a performance penalty
        #pyplot.close()
        
        pyplot.figure()
        pyplot.plot(xCoords, yCoords, '-+r')
        pyplot.title('Tooth nr: ' + str((j+1)))
        pyplot.xlabel('x\'')
        pyplot.ylabel('y\'')
        for i in range(X.shape[0]):
            xCs, yCs = mu.extract_coordinates(Y[i,:])
            pyplot.plot(xCs, yCs, '*g')
    
        pyplot.xlabel('x\'')
        pyplot.ylabel('y\'')
        pyplot.gca().invert_yaxis()
        pyplot.axis('equal')
        fname = c.get_fname_vis_pa((j+1), samples_included=True)
        pyplot.savefig(fname, bbox_inches='tight')
        #You get a runtime warning if you open more than 20 figures
        #Closing comes with a performance penalty
        #pyplot.close()
            
def plot_mean(M, nr_tooth=1, closed_curve=False):
    '''
    Plots the landmarks corresponding to the mean shape
    in the model coordinate frame.
    @param M:                   the mean shape in the model coordinate frame
    @param nr_tooth:            the number of the tooth
                                (just used for the title of the plot)
    @param closed_curve:        must the curve be closed
    '''
    xCoords, yCoords = mu.extract_coordinates(M)
    if (closed_curve):
        xCoords = mu.make_circular(xCoords)
        yCoords = mu.make_circular(yCoords)
    
    pyplot.figure(2)
    # x coordinates , y coordinates
    pyplot.plot(xCoords, yCoords, '-+r')
    pyplot.title('Tooth nr: ' + str(nr_tooth))
    pyplot.xlabel('x\'')
    pyplot.ylabel('y\'')
    pyplot.gca().invert_yaxis()
    pyplot.axis('equal')
    pyplot.show()
    
def plot_all(M, X, nr_tooth=1, closed_curve=False):
    '''
    Plots the landmarks corresponding to the mean shape
    together with all the training samples in the model
    coordinate frame
    @param M:                   the mean shape in the model coordinate frame
    @param X:                   the training samples in the model coordinate frame
    @param nr_tooth:            the number of the tooth
                                (just used for the title of the plot)
    @param closed_curve:        must the curve be closed
    '''
    xCoords, yCoords = mu.extract_coordinates(M)
    if (closed_curve):
        xCoords = mu.make_circular(xCoords)
        yCoords = mu.make_circular(yCoords)
    
    pyplot.figure(2)
    # x coordinates , y coordinates
    pyplot.plot(xCoords, yCoords, '-+r')
    pyplot.title('Tooth nr: ' + str(nr_tooth))
    pyplot.xlabel('x\'')
    pyplot.ylabel('y\'')
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
    #plot_all(M, Y, closed_curve=True)
    store_plotted_means(closed_curve=True)