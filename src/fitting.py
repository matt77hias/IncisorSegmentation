'''
Single-Resolution Active Shape Models' fitting procedure.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import configuration as c
import loader as l
import numpy as np
import cv2

import math
import math_utils as mu
import procrustes_analysis as pa
import principal_component_analysis as pca
import fitting_function as ff

MS = None                       #MS contains for each tooth, the tooth model (in the model coordinate frame)
EWS = []                        #EWS contains for each tooth, a (Eigenvalues, Eigenvectors) pair (in the model coordinate frame)
fs = None                       #fitting function for each tooth, for each landmark.

offsetY = 497.0                 #The landmarks refer to the non-cropped images, so we need the vertical offset (up->down)
                                #to locate them on the cropped images.
offsetX = 1234.0                #The landmarks refer to the non-cropped images, so we need the horizontal offset (left->right)
                                #to locate them on the cropped images.

k = 5                           #The number of pixels to sample either side for each of the model points along the profile normal
                                #(used for creating the fitting functions)
m = 10                          #The number of pixels to sample either side for each of the model points along the profile normal
                                #(used while iterating)
method='SCD'                    #The method used for preproccesing.

convergence_threshold = 0.0001  #The convergence threshold (used while iterating).
tolerable_deviation = 3         #The number of deviations that are tolerable by the models (used for limiting the shape).

def fit_all_teeth(img, PS):
    '''
    Fits all the teeth in the given image.
    @param img:             the image  
    @param PS:              the start points for each tooth
    '''
    for j in range(PS.shape[0]):
        fit_tooth(img, PS[j], j)

def fit_tooth(img, P, tooth_index, show=False):
    '''
    Fits all the tooth corresponding to the given tooth index in the given image.
    @param img:             the image  
    @param P:               the start points for the target tooth
    @param tooth_index:     the index of the the target tooth (used in MS, EWS, fs)
    @param show:            must the intermediate results (after each iteration) be displayed
    '''
    gradient = ff.create_gradient(img)
    nb_tests = 2*(m-k)+1
    
    if (show): 
        show_interation(np.copy(img), 0, P)
        cv2.waitKey(0)
    
    nb_it = 1
    convergence = False
    while (not convergence) :
        pxs, pys = mu.extract_coordinates(P)
        for i in range(c.get_nb_landmarks()):
            Gi, Coords = ff.create_Gi(gradient, m, i, pxs, pys)
            f_optimal = fs[tooth_index][i](ff.normalize_Gi(Gi[0:2*k+1]))
            c_optimal = k
            for t in range(1,nb_tests):
                f = fs[tooth_index][i](ff.normalize_Gi(Gi[t:t+2*k+1]))
                if f < f_optimal:
                    f_optimal = f
                    c_optimal = t+k
            pxs[i] = Coords[(2*c_optimal)] 
            pys[i] = Coords[(2*c_optimal+1)]
        
        P_new = validate(tooth_index, mu.zip_coordinates(pxs, pys))
        if (np.linalg.norm(P-P_new) < convergence_threshold): convergence = True    
        
        P = P_new
        
        if (show): 
            show_interation(np.copy(img), nb_it, P)
            cv2.waitKey(0)
        
        nb_it += 1
                
def validate(tooth_index, P):
    '''
    Validates the current points P for the target tooth corresponding to the given
    tooth index.
    @param P:               the current points for the target tooth
    @param tooth_index:     the index of the the target tooth (used in MS, EWS, fs)
    '''
    MU = MS[tooth_index]
    E, W = EWS[tooth_index]

    xm, ym = mu.get_center_of_gravity(P)
    tx, ty, s, theta = mu.full_align_params(P, MU)
    PY = mu.full_align(P, tx, ty, s, theta)
    
    bs = pca.project(W, PY, MU)
    for i in range(E.shape[0]):
        b_min = -tolerable_deviation*math.sqrt(E[i])
        b_max =  tolerable_deviation*math.sqrt(E[i])
        b = bs[i]
        
        print(str(b_min) + ' # ' + str(b) + ' # ' + str(b_max))
        
        if b < b_min: 
            bs[i] = b_min
            print ('min')
        if b > b_max: 
            bs[i] = b_max
            print ('max')

    PY = pca.reconstruct(W, bs, MU)
    P = mu.full_align(PY, xm, ym, 1.0 / s, -theta)
    return P
    
def show_interation(img, nb_it, P, color_init=np.array([0,255,255]), color_mid=np.array([255,0,255]), color_end=np.array([255,255,0]), color_line=np.array([255,0,0])):
    '''
    Displays the current points markes on the given image.
    @param img:         the image
    @param nb_it:       the number of this iteration
    @param P:           the current points for the target tooth
    @param color_init:  the BGR color for the first landmark 
    @param color_mid:   the BGR color for all landmarks except the first and last landmark
    @param color_end:   the BGR color for the last landmark
    @param color_line:  the BGR color for the line between two consecutive landmarks
    '''
    xs, ys = mu.extract_coordinates(P)  
    for k in range(c.get_nb_landmarks()):
        x = int(xs[k])
        y = int(ys[k])
        if (k == c.get_nb_landmarks()-1):
            x_succ = int(xs[0])
            y_succ = int(ys[0])
        else:
            x_succ = int(xs[(k+1)])
            y_succ = int(ys[(k+1)])
        cv2.line(img, (x,y), (x_succ,y_succ), color_line)
    
    for k in range(c.get_nb_landmarks()):
        x = int(xs[k])
        y = int(ys[k])
        if (k == 0):
            img[y,x] = color_init
        elif (k == c.get_nb_landmarks()-1):
            img[y,x] = color_end
        else:
            img[y,x] = color_mid
    
    txt = 'Iteration: ' + str(nb_it)
    cv2.imshow(txt, img)

def preprocess(trainingSamples):
    '''
    Creates MS, EWS and fs, used by the fitting procedure
        * MS contains for each tooth, the tooth model (in the model coordinate frame)
        * EWS contains for each tooth, a (Eigenvalues, Eigenvectors) pair (in the model coordinate frame)
        * fitting function for each tooth, for each landmark.
    '''
    global MS, EWS, fs
    XS = l.create_partial_XS(trainingSamples)
    MS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    
    for j in range(c.get_nb_teeth()):
        M, Y = pa.PA(XS[j,:,:])
        MS[j,:] = M
        E, W, MU = pca.pca_percentage(Y)
        EWS.append((E, W))

    GS = ff.create_partial_GS(trainingSamples, XS, MS, offsetX=offsetX, offsetY=offsetY, k=k, method=method)
    fs = ff.create_fitting_functions(GS)
    
def original_to_cropped(P):
    '''
    Crops the given points. Used when working with non-cropped initial target points.
    The whole fitting procdure itself doesn't work with offsets at all.
    @pre    The coordinates are stored as successive xi, yi, xj, yj, ...
    @param  P:   the points to crop
    @return The cropped version of P.
    '''
    for i in range(P.shape[0] / 2):
        P[(2*i)] -= offsetX
        P[(2*i+1)] -= offsetY
    return P
    
if __name__ == "__main__":
    trainingSamples = range(2, (c.get_nb_trainingSamples()+1))
    preprocess(trainingSamples)
    
    fname = c.get_fname_vis_pre(1, 'SCD')
    img = cv2.imread(fname)
    
    #fname = c.get_fname_fitting_manual_landmark(1, 1)
    #P = original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))
    fname = c.get_fname_original_landmark(1, 1)
    P = original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))
    
    fit_tooth(img, P, 0, show=True)
 

    #to separate .py
    
    #SX = click.draw_landmarks(c.get_fname_vis_pre(1, 'SCD'))
    #nr_tooth = 2
    #nr_trainingSample = 1
    #fname = c.get_fname_fitting_manual_landmark(nr_trainingSample, nr_tooth)
    #SX = np.fromfile(fname, dtype=float, count=-1, sep=" ")   
    
    
    