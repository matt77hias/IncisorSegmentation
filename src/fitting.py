import click
import configuration as c
import loader as l
import math_utils as mu
import numpy as np
import procrustes_analysis as pa
import principal_component_analysis as pca


def evaluate(W, SY, mu):
    C = pca.project(W, SY, mu)
    Reconstruction = pca.reconstruct(W, C, mu)
    return np.linalg.norm(SY - Reconstruction) 
        
if __name__ == "__main__":
    M, Y = pa.PA(l.create_full_X(nr_tooth=1))
    E, W, MU = pca.pca_percentage(Y)
    
    #SX = click.draw_landmarks(c.get_fname_vis_pre(1, 'SCD'))
    nr_tooth = 2
    nr_trainingSample = 1
    fname = c.get_fname_fitting_manual_landmark(nr_trainingSample, nr_tooth)
    SX = np.fromfile(fname, dtype=float, count=-1, sep=" ")   
    
    ST = mu.center_onOrigin(SX)
    SY = mu.align_with(ST, M)
    
    print(evaluate(W, SY, MU))
    