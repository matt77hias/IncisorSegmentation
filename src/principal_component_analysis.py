# -*- coding: utf-8 -*-
'''
Principal Component Analysis
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import numpy as np
import math_utils as mu

def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    @param W:               the eigenvectors
    @param X:               the image to project
    @param mu:              the average image
    @return The projection of X on the space spanned by the vectors in W.
    '''
    return np.dot(W.T, (X-mu))

def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    @param W:               the eigenvectors
    @param Y:               the PCA-coefficients
    @param mu:              the average image
    @return The reconstruction of an image based on its PCA-coefficients, the eigenvectors and
            the average mu.
    '''
    return (np.dot(W, Y) + mu)

def pca(X, nb_components=0):
    '''
    Do a PCA (Principal Component Analysis) on X
    @param X:                np.array containing the training samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return The nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    [n,d] = X.shape
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n
    
    #Turn a set of possibly correlated variables into a smaller set of uncorrelated variables.
    #The idea is, that a high-dimensional dataset is often described by correlated variables and
    #therefore only a few meaningful dimensions account for most of the information.
    #The PCA method finds the directions with the greatest variance in the data, called principal components.
    
    MU = X.mean(axis=0)
    for i in range(n):
        X[i,:] -= MU
    
    S = (np.dot(X, X.T) / float(n))
    eigenvalues, eigenvectors = np.linalg.eig(S)
    
    #And about the negative eigenvalues, it is just a matter of eigh.
    #As eigenvalues shows the variance in a direction, we care about absolute
    #value but if we change a sign, we also have to change the "direcction" (eigenvector).
    #You can make this multiplying negative eigenvalues and their corresponding eigenvectors with -1.0
    s = np.where(eigenvalues < 0)
    eigenvalues[s] = eigenvalues[s] * -1.0
    eigenvectors[:,s] = eigenvectors[:,s] * -1.0

    #The nb_components largest eigenvalues and eigenvectors of the covariance matrix
    indexes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indexes][0:nb_components]
    eigenvectors = eigenvectors[:,indexes][:,0:nb_components]

    eigenvectors = np.dot(X.T, eigenvectors)
    for i in range(nb_components):
        eigenvectors[:,i] = mu.normalize_vector(eigenvectors[:,i])
    
    return (eigenvalues, eigenvectors, MU)