'''
Single-Resolution Active Shape Models' fitting procedure.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import configuration as c
import loader as l
import numpy as np
import cv2
import math_utils as mu
import procrustes_analysis as pa
import principal_component_analysis as pca
import fitting_function as ff

from matplotlib import pyplot

MS = None                       #MS contains for each tooth, the tooth model (in the model coordinate frame)
EWS = []                        #EWS contains for each tooth, a (sqrt(Eigenvalues), Eigenvectors) pair (in the model coordinate frame)
fns = None                      #fitting functions for each tooth, for each landmark, for the profile normal through that landmark.
fts = None                      #fitting functions for each tooth, for each landmark, for the profile gradient through that landmark.

offsetY = 497.0                 #The landmarks refer to the non-cropped images, so we need the vertical offset (up->down)
                                #to locate them on the cropped images.
offsetX = 1234.0                #The landmarks refer to the non-cropped images, so we need the horizontal offset (left->right)
                                #to locate them on the cropped images.

k = 5                           #The number of pixels to sample either side for each of the model points along the profile normal
                                #(used for creating the fitting functions)
m = 8                           #The number of pixels to sample either side for each of the model points along the profile normal
                                #(used while iterating)
method='SCD'                    #The method used for preproccesing.

convergence_threshold = 0.002   #The convergence threshold (used while iterating).
tolerable_deviation = 3         #The number of deviations that are tolerable by the models (used for limiting the shape).

def fit_tooth(img, P, tooth_index, fitting_function=0, show=False):
    '''
    Fits the tooth corresponding to the given tooth index in the given image.
    @param img:                 the image  
    @param P:                   the start points for the target tooth
    @param tooth_index:         the index of the the target tooth (used in MS, EWS, fs)
    @param fitting_function:    the fitting function used
                                * 0: fitting function along profile normal through landmark +
                                     fitting function along profile gradient through landmark
                                * 1: fitting function along profile normal through landmark
                                * 2: fitting function along profile gradient through landmark
    @param show:                must the intermediate results (after each iteration) be displayed
    @return The fitted points for the tooth corresponding to the given tooth index
            and the number of iterations used.
    '''
    nb_it = 0
    convergence = False
    while (not convergence) :
        nb_it += 1
        pxs, pys = mu.extract_coordinates(P)
        for i in range(c.get_nb_landmarks()):
            tx, ty, nx, ny = ff.create_ricos(img, i, pxs, pys)
            f_optimal = float("inf")
            for n in range(-(m-k), (m-k)+1):
                for t in range(-(m-k), (m-k)+1):
                    x = round(pxs[i] + n * nx + t * tx)
                    y = round(pys[i] + n * ny + t * ty)
                    fn = fns[tooth_index][i](ff.normalize_Gi(ff.create_Gi(img, k, x, y, nx, ny)))
                    ft = fts[tooth_index][i](ff.normalize_Gi(ff.create_Gi(img, k, x, y, tx, ty)))
                    f = evaluate_fitting(fn=fn, ft=ft, fitting_function=fitting_function)
                    if f < f_optimal:
                        f_optimal = f
                        cx = x
                        cy = y
            pxs[i] = cx
            pys[i] = cy
        
        P_new = validate(img, tooth_index, mu.zip_coordinates(pxs, pys), nb_it, show)
        conv  = np.linalg.norm(P-P_new)/np.linalg.norm(P)
        #if (conv < convergence_threshold): convergence = True 
        
        tx_old, ty_old, s_old, theta_old = mu.full_align_params(P, MS[tooth_index])
        tx, ty, s, theta = mu.full_align_params(P_new, MS[tooth_index])
        xm_old, ym_old = mu.get_center_of_gravity(P)
        xm, ym = mu.get_center_of_gravity(P_new)
        diff_tx = abs(xm_old - xm)
        diff_ty = abs(ym_old - ym)
        diff_s = abs(s_old - s)
        diff_theta = abs(theta_old - theta)
        print(str(diff_tx) + ' # ' + str(diff_ty) + ' # ' + str(diff_s) + ' # ' + str(diff_theta))
        
        P = P_new
        if (nb_it >= 30): convergence = True 
        
    print('Fitting number of iterations: ' + str(nb_it))
    return P, nb_it
    
def evaluate_fitting(fn=0, ft=0, fitting_function=0):
    ''''
    Evaluates the fitting function.
    @param fn:                  the fitting function evaluated along the profile normal
    @param ft:                  the fitting function evaluated along the profile gradient
    @param fitting_function:    the fitting function used
                                * 0: fitting function along profile normal through landmark +
                                     fitting function along profile gradient through landmark
                                * 1: fitting function along profile normal through landmark
                                * 2: fitting function along profile gradient through landmark
    @return The evaluated fitting function.
    '''
    if (fitting_function==0):
        return fn + ft
    elif (fitting_function==1):
        return fn
    elif (fitting_function==1):
        return ft
    else:
        return 0
                
def validate(img, tooth_index, P_before, nb_it, show=False):
    '''
    Validates the current points P for the target tooth corresponding to the given
    tooth index.
    @param img:             the image
    @param P_before:        the current points for the target tooth before validation
                            in the image coordinate frame
    @param tooth_index:     the index of the the target tooth (used in MS, EWS, fs)
    @param nb_it:           the number of this iteration
    @param show:            must the intermediate results (after each iteration) be displayed
    '''
    MU = MS[tooth_index]
    E, W = EWS[tooth_index]

    xm, ym = mu.get_center_of_gravity(P_before)
    tx, ty, s, theta = mu.full_align_params(P_before, MU)
    PY_before = mu.full_align(P_before, tx, ty, s, theta)
    
    bs = pca.project(W, PY_before, MU)
    bs = np.maximum(np.minimum(bs, tolerable_deviation*E), -tolerable_deviation*E)

    PY_after = pca.reconstruct(W, bs, MU)
    P_after = mu.full_align(PY_after, xm, ym, 1.0 / s, -theta)
    
    if (show): 
        show_validation(nb_it, PY_before, PY_after, tooth_index)
        show_interation(np.copy(img), nb_it, P_before, P_after)
        cv2.waitKey(0)
        pyplot.close()  

    return P_after
    
def show_validation(nb_it, PY_before, PY_after, tooth_index):
    '''
    Plots the landmarks corresponding to the mean shape in the model coordinate frame
    and the landmarks corresponding to the current points in the model coordinate frame
    @param nb_it:           the number of this iteration
    @param PY_before:       the current points for the target tooth before validation
                            in the model coordinate frame
    @param PY_after:        the current points for the target tooth after validation
                            in the model coordinate frame
    @param tooth_index:     the index of the the target tooth (used in MS, EWS, fs)
    '''
    mxs, mys = mu.extract_coordinates(MS[tooth_index,:])
    mxs = mu.make_circular(mxs)
    mys = mu.make_circular(mys)
    rxs, rys = mu.extract_coordinates(PY_before)
    rxs = mu.make_circular(rxs)
    rys = mu.make_circular(rys)
    gxs, gys = mu.extract_coordinates(PY_after)
    gxs = mu.make_circular(gxs)
    gys = mu.make_circular(gys)
    
    pyplot.figure(1)
    # x coordinates , y coordinates
    pyplot.plot(mxs, mys, '-+b')
    pyplot.plot(rxs, rys, '-+r')
    pyplot.plot(gxs, gys, '-+g')
    txt = 'Model Coordinate Frame - Iteration: ' + str(nb_it)
    pyplot.title(txt)
    pyplot.xlabel('x\'')
    pyplot.ylabel('y\'')
    pyplot.gca().invert_yaxis()
    pyplot.axis('equal')
    pyplot.show()
    
def show_interation(img, nb_it, P_before, P_after, color_init=np.array([0,255,255]), color_mid=np.array([255,0,255]), color_end=np.array([255,255,0]), color_line_before=np.array([0,0,255]), color_line_after=np.array([0,255,0])):
    '''
    Displays the current points markes on the given image.
    This method displays the movement in the image coordinate frame.
    @param img:                 the image
    @param nb_it:               the number of this iteration
    @param P_before:            the current points for the target tooth before validation
                                in the image coordinate frame
    @param P_after:             the current points for the target tooth after validation
                                in the image coordinate frame
    @param color_init:          the BGR color for the first landmark 
    @param color_mid:           the BGR color for all landmarks except the first and last landmark
    @param color_end:           the BGR color for the last landmark
    @param color_line_before:   the BGR color for the line between two consecutive landmarks
    @param color_line_after:    the BGR color for the line between two consecutive landmarks
    '''
    rxs, rys = mu.extract_coordinates(P_before)
    gxs, gys = mu.extract_coordinates(P_after)
    for k in range(c.get_nb_landmarks()):
        rx = int(rxs[k])
        ry = int(rys[k])
        gx = int(gxs[k])
        gy = int(gys[k])
        if (k == c.get_nb_landmarks()-1):
            rx_succ = int(rxs[0])
            ry_succ = int(rys[0])
            gx_succ = int(gxs[0])
            gy_succ = int(gys[0])
        else:
            rx_succ = int(rxs[(k+1)])
            ry_succ = int(rys[(k+1)])
            gx_succ = int(gxs[(k+1)])
            gy_succ = int(gys[(k+1)])
        cv2.line(img, (rx,ry), (rx_succ,ry_succ), color_line_before)
        cv2.line(img, (gx,gy), (gx_succ,gy_succ), color_line_after)
    
    for k in range(c.get_nb_landmarks()):
        rx = int(rxs[k])
        ry = int(rys[k])
        gx = int(gxs[k])
        gy = int(gys[k])
        if (k == 0):
            img[ry,rx] = color_init
            img[gy,gx] = color_init
        elif (k == c.get_nb_landmarks()-1):
            img[ry,rx] = color_end
            img[gy,gx] = color_end
        else:
            img[ry,rx] = color_mid
            img[gy,gx] = color_mid
    
    txt = 'Image Coordinate Frame - Iteration: ' + str(nb_it)
    #cv2.imshow(txt, img)
    return img

def preprocess(trainingSamples):
    '''
    Creates MS, EWS and fs, used by the fitting procedure
        * MS contains for each tooth, the tooth model (in the model coordinate frame)
        * EWS contains for each tooth, a (sqrt(Eigenvalues), Eigenvectors) pair (in the model coordinate frame)
        * fitting function for each tooth, for each landmark.
    '''
    global MS, EWS, fns, fts
    XS = l.create_partial_XS(trainingSamples)
    MS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    
    for j in range(c.get_nb_teeth()):
        M, Y = pa.PA(XS[j,:,:])
        MS[j,:] = M
        E, W, MU = pca.pca_percentage(Y)
        EWS.append((np.sqrt(E), W))

    GNS, GTS = ff.create_partial_GS(trainingSamples, XS, MS, offsetX=offsetX, offsetY=offsetY, k=k, method=method)
    fns, fts = ff.create_fitting_functions(GNS, GTS)
    
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
    
def test():
    for i in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(i)
        preprocess(trainingSamples)
        
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        
        for j in range(c.get_nb_teeth()):
            fname = c.get_fname_original_landmark(i, (j+1))
            P = original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))
            R, nb_it = fit_tooth(img, P, j)
            fname = str(i) + '-' + str((j+1)) + '#' + str(nb_it) + '.png'
            cv2.imwrite(fname, show_interation(np.copy(img), nb_it, P, R))
    
if __name__ == "__main__":
    test()

    #to separate .py
    
    #SX = click.draw_landmarks(c.get_fname_vis_pre(1, 'SCD'))
    #nr_tooth = 2
    #nr_trainingSample = 1
    #fname = c.get_fname_fitting_manual_landmark(nr_trainingSample, nr_tooth)
    #SX = np.fromfile(fname, dtype=float, count=-1, sep=" ")   