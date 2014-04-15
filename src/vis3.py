'''
Contains some visualization functions for displaying the results
of the Principal Component Analysis
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import numpy as np
import pylab

import math_utils as mu
import loader as l
import procrustes_analysis as pa
import principal_component_analysis as pca

if __name__ == '__main__':
    X = l.create_full_X(nr_tooth=1)
    M, Y = pa.PA(X)
    eigenvalues, eigenvectors, MU = pca.pca_percentage(Y)
