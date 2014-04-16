'''
Loader for training samples
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import numpy as np
import configuration as c

def create_full_X(nr_tooth=1):
    '''
    Creates an array that contains all the training samples
    corresponding to the given tooth number.
    @param nrTooth:              the number of the tooth
    @return np.array, shape=(nb of training samples, nb of dimensions)
    '''
    return create_partial_X(c.get_trainingSamples_range(), nr_tooth)

def create_partial_X(trainingSamples, nr_tooth=1):
    '''
    Creates an array that contains all the training samples
    corresponding to the given training samples and tooth number.
    @param trainingSamples:      the training samples
    @param nrTooth:              the number of the tooth
    @return np.array, shape=(nb of training samples, nb of dimensions)
    '''
    X = np.zeros((len(trainingSamples), c.get_nb_dim()))
    for i in trainingSamples:
        fname = c.get_fname_original_landmark(i, nr_tooth)
        print("loaded: " + fname)
        X[(i-1),:] = np.fromfile(fname, dtype=float, count=-1, sep=' ')
    return X