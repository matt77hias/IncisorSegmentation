# -*- coding: utf-8 -*-
'''
Procrustes Analysis
Aligning a set of training shapes into a common coordinate frame
because the shape of an object is normally considered independent
of the position, orientation and scale of that object.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''
import numpy as np

import math_utils as mu

convergence_threshold = 0.01

def PA(X):
    '''
    Do a PA (Procrustes Analysis) on X
    @param X:                np.array containing the training samples
                             shape = (nb samples, nb dimensions of each sample)
    @return The mean shape and training samples in the model coordinate frame.
    '''
    #Translation
    XT = translate(X)
    #Initial estimate of mean shape, rescale
    X0 = M = mu.normalize_vector(XT[0,:])
    
    #Align all the shapes with the current estimate of the mean shape
    Y = np.zeros(XT.shape)
    Y[0,:] = M
    for i in range(1, XT.shape[0]):
        Y[i,:] = mu.align_with(XT[i,:], X0)
          
    #Re-estimate the mean from aligned shapes
    MN = Y.mean(axis=0)
    #Apply constraints on scale and orientation to the current estimate
    #of the mean by aliging it with X0 and scaling so that |M|=1
    MN = mu.align_with(MN, X0)
    MN = mu.normalize_vector(MN)
    
    #Iterative approach
    it = 1
    while (not is_converged(M, MN)):
        M = MN
        for i in range(XT.shape[0]):
            Y[i,:] = mu.align_with(XT[i,:], M)
        MN = Y.mean(axis=0)
        MN = mu.align_with(MN, X0)
        MN = mu.normalize_vector(MN)
        it += 1
        
    print("PA number of iterations: " + str(it))
    return MN, Y
    
def is_converged(M, MN):
    '''
    Checks if the mean shape is converged.
    @param  M:         the previous mean shape
    @param  MN:        the new mean shape
    @return True if and only if the mean shape is converged.
    '''
    return ((M - MN) < (convergence_threshold * np.ones(M.shape))).all()

def translate(X):
    '''
    Translates the given training samples so that their centre of gravity is at the origin.
    This translation is done for each training sample.
    @param  X:        the training samples
    @return The translated training samples with their centre of gravity at the origin.
    '''
    XT = np.zeros(X.shape)
    for i in range(X.shape[0]):
        XT[i,:] = mu.centre_onOrigin(X[i,:])
    return XT