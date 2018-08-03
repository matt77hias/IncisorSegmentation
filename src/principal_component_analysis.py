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

def pca_nb(X, nb_components=0):
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
        
    eigenvalues, eigenvectors, MU = pca_raw(X)

    print("PCA number of components: " + str(nb_components))

    #The nb_components largest eigenvalues and eigenvectors of the covariance matrix
    eigenvalues = eigenvalues[0:nb_components]
    eigenvectors = eigenvectors[:,0:nb_components]

    return (eigenvalues, eigenvectors, MU)
    
def pca_percentage(X, percentage=0.98):
    '''
    Do a PCA (Principal Component Analysis) on X
    @param X:                np.array containing the training samples
                             shape = (nb samples, nb dimensions of each sample)
    @param percentage:       the proportion of the variance that must be taken into
                             account
    @return The eigenvalues and eigenvectors of the covariance matrix that explain 
            the proportion 'percentage' of the variance exibited in the training set
            and return the average sample 
    '''
    #n = X.shape[0]
    if (percentage <= 0) or (percentage>1):
        percentage = 0.98
        
    eigenvalues, eigenvectors, MU = pca_raw(X)
    
    s = sum(eigenvalues)
    cs = nb_components = 0
    for i in range(eigenvalues.shape[0]):
        cs += (eigenvalues[i] / s)
        nb_components += 1
        if cs > percentage:
            break
    
    print("PCA number of components: " + str(nb_components)) 

    #The nb_components largest eigenvalues and eigenvectors of the covariance matrix
    eigenvalues = eigenvalues[0:nb_components]
    eigenvectors = eigenvectors[:,0:nb_components]

    return (eigenvalues, eigenvectors, MU)
    
def pca_raw(X):
    '''
    Do a PCA (Principal Component Analysis) on X
    @param X:                np.array containing the training samples
                             shape = (nb samples, nb dimensions of each sample)
    @return ALL the eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    n = X.shape[0]
    
    #Turn a set of possibly correlated variables into a smaller set of uncorrelated variables.
    #The idea is, that a high-dimensional dataset is often described by correlated variables and
    #therefore only a few meaningful dimensions account for most of the information.
    #The PCA method finds the directions with the greatest variance in the data, called principal components.
    
    MU = X.mean(axis=0)
    for i in range(n):
        X[i,:] -= MU
    
    S = (np.dot(X, X.T) / float(n))
    eigenvalues, eigenvectors = np.linalg.eig(S)
    
    #About the negative eigenvalues, it is just a matter of eig(h).
    #As eigenvalues show the variance in a direction, we care about absolute
    #value but if we change a sign, we also have to change the "direction" (eigenvector).
    #You can make this multiplying negative eigenvalues and their corresponding eigenvectors with -1.0
    s = np.where(eigenvalues < 0)
    eigenvalues[s] = eigenvalues[s] * -1.0
    eigenvectors[:,s] = eigenvectors[:,s] * -1.0

    #All the eigenvalues and eigenvectors of the covariance matrix
    indexes = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indexes]
    eigenvectors = eigenvectors[:,indexes]

    eigenvectors = np.dot(X.T, eigenvectors)
    for i in range(n):
        eigenvectors[:,i] = mu.normalize_vector(eigenvectors[:,i])
    
    return (eigenvalues, eigenvectors, MU)
