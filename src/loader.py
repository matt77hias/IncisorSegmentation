'''
Loader for training samples
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import numpy as np
import configuration as c

def create_full_XS():
    '''
    Creates an array that contains all the training samples
    corresponding to all the teeth.
    @return np.array, shape=(nb of teeth, nb of training samples, nb of dimensions)
    '''
    return create_partial_XS(c.get_trainingSamples_range())
    
def create_partial_XS(trainingSamples):
    '''
    Creates an array that contains all the training samples
    corresponding to the given training samples and corresponding to all the teeth.
    @param trainingSamples:      the training samples
    @return np.array, shape=(nb of teeth, nb of training samples, nb of dimensions)
    '''
    XS = np.zeros(np.array([c.get_nb_teeth(), len(trainingSamples), c.get_nb_dim()]))
    for j in range(c.get_nb_teeth()):
        XS[j,:,:] = create_partial_X(trainingSamples, nr_tooth=(j+1))
    return XS

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
    index = 0
    for i in trainingSamples:
        fname = c.get_fname_original_landmark(i, nr_tooth)
        print("loaded: " + fname)
        X[index,:] = np.fromfile(fname, dtype=float, count=-1, sep=' ')
        index += 1
    return X