
import configuration as c
import loader as l
import numpy as np

import procrustes_analysis as pa
import principal_component_analysis as pca
import fitting_function as fit

XS = None
MS = None
EWS = []
fs = None

offsetY = 497.0
offsetX = 1234.0
k = 5
method='SCD'


def preprocess(trainingSamples):
    global XS, MS, EWS, fs
    XS = l.create_partial_XS(trainingSamples)
    MS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    
    for j in range(c.get_nb_teeth()):
        M, Y = pa.PA(l.create_full_X(j+1))
        MS[j,:] = M
        MU, E, W = pca.pca_percentage(Y)
        EWS.append((E, W))

    GS = fit.create_partial_GS(trainingSamples, XS, MS, offsetX=offsetX, offsetY=offsetY, k=k, method=method)
    fs = fit.create_fitting_functions(GS)
    
if __name__ == "__main__":
    preprocess(c.get_trainingSamples_range())

        

    #to separate .py
    
    #SX = click.draw_landmarks(c.get_fname_vis_pre(1, 'SCD'))
    #nr_tooth = 2
    #nr_trainingSample = 1
    #fname = c.get_fname_fitting_manual_landmark(nr_trainingSample, nr_tooth)
    #SX = np.fromfile(fname, dtype=float, count=-1, sep=" ")   
    
    
    