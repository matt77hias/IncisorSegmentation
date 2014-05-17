
import configuration as c
import loader as l
import numpy as np
import cv2

import math_utils as mu
import procrustes_analysis as pa
import principal_component_analysis as pca
import fitting_function as ff

XS = None
MS = None
EWS = []
fs = None

offsetY = 497.0
offsetX = 1234.0
k = 5
m = 10
method='SCD'


def fit(P, nr_sample, method=''):
    fname = c.get_fname_vis_pre(nr_sample, method)
    img = cv2.imread(fname)
    gradient = ff.create_gradient(img)
    
    nb_tests = 2*(m-k)+1
    pxs, pys = mu.extract_coordinates(P)
    
    for i in range(c.get_nb_landmarks()):
        Gi = ff.create_Gi(img, m, i, pxs, pys, offsetX, offsetY)
        for t in range(nb_tests):
           Gi[t:t+2*k+1]
        
    
    
    
    
    
    

def preprocess(trainingSamples):
    global XS, MS, EWS, fs
    XS = l.create_partial_XS(trainingSamples)
    MS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    
    for j in range(c.get_nb_teeth()):
        M, Y = pa.PA(l.create_full_X(j+1))
        MS[j,:] = M
        MU, E, W = pca.pca_percentage(Y)
        EWS.append((E, W))

    GS = ff.create_partial_GS(trainingSamples, XS, MS, offsetX=offsetX, offsetY=offsetY, k=k, method=method)
    fs = ff.create_fitting_functions(GS)
    
if __name__ == "__main__":
    preprocess(c.get_trainingSamples_range())

        

    #to separate .py
    
    #SX = click.draw_landmarks(c.get_fname_vis_pre(1, 'SCD'))
    #nr_tooth = 2
    #nr_trainingSample = 1
    #fname = c.get_fname_fitting_manual_landmark(nr_trainingSample, nr_tooth)
    #SX = np.fromfile(fname, dtype=float, count=-1, sep=" ")   
    
    
    