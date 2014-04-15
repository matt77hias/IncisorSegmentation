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
    #Translation
    translate(X)
    #Initial estimate of mean shape, rescale
    M = mu.normalize_vector(X[0,:])
    X0 = M
    
    #Align all the shapes with the current estimate of the mean shape
    Y = np.zeros(X.shape)
    Y[0,:] = M
    for i in range(1, X.shape[0]):
        Y[i,:] = mu.align_with(X[i,:], X0)
    
    #Re-estimate the mean from aligned shapes
    MN = Y.mean(axis=0)
    #Apply constraints on scale and orientation to the current estimate
    #of the mean by aliging it with X0 and scaling so that |M|=1
    MN = mu.align_with(MN, X0)
    MN = mu.normalize_vector(MN)
    
    #Iterative approach
    while (not is_converged(M, MN)):
        M = MN
        for i in range(X.shape[0]):
            Y[i,:] = mu.align_with(X[i,:], M)
        MN = Y.mean(axis=0)
        MN = mu.align_with(MN, X0)
        MN = mu.normalize_vector(MN)
    
    return MN
    
def is_converged(M, MN):
    return ((M - MN) < (convergence_threshold * np.ones(M.shape))).all()

def translate(X):
    '''
    Translates the given training samples so that their centre of gravity is at the origin.
    This translation is done for each training sample.
    @param  X:     the training samples
    @return The translated training samples with their centre of gravity at the origin.
    '''
    for i in range(X.shape[0]):
        X[i,:] -= np.mean(X[i,:])
    return X