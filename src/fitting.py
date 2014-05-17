
import configuration as c
import loader as l
import numpy as np
import cv2

import math
import math_utils as mu
import procrustes_analysis as pa
import principal_component_analysis as pca
import fitting_function as ff

MS = None
EWS = []
fs = None

offsetY = 497.0
offsetX = 1234.0
k = 5
m = 10
method='SCD'

convergence_threshold = 0.00001
tolerable_deviation = 3

def fit(img, P, tooth_index):
    gradient = ff.create_gradient(img)
    nb_tests = 2*(m-k)+1
    pxs, pys = mu.extract_coordinates(P)
    
    convergence = False
    while (not convergence) :
        for i in range(c.get_nb_landmarks()):
            Gi, Coords = ff.create_Gi(gradient, m, i, pxs, pys, offsetX, offsetY)
            f_optimal = np.linalg.norm(fs[tooth_index,i](mu.normalize(Gi[0:2*k+1])))
            c_optimal = k
            for t in range(1,nb_tests):
                f = np.linalg.norm(fs[tooth_index,i](mu.normalize(Gi[t:t+2*k+1])))
                if f < f_optimal:
                    f_optimal = f
                    c_optimal = t+k
            pxs[i] = Coords[(2*c_optimal)] 
            pys[i] = Coords[(2*c_optimal+1)]
        
        P_new = validate(tooth_index, mu.zip_coordinates(pxs, pys))
        if (np.linalg.norm(P-P_new) < convergence_threshold): convergence = True    
                
def validate(tooth_index, P):
    MU = MS[tooth_index]
    E, W = EWS[tooth_index]
    bs = pca.project(W, P, MU)

    for i in range(E.shape[0]):
        b_min = -tolerable_deviation*math.sqrt(E[i])
        b_max =  tolerable_deviation*math.sqrt(E[i])
        b = bs[i]
        if b < b_min: bs[i] = b_min     #TODO: more robust limitations
        elif b > b_max: bs[i] = b_max   #TODO: more robust limitations

    return pca.reconstruct(W, bs, MU)

def preprocess(trainingSamples):
    global MS, EWS, fs
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
    
    
    