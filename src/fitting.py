'''
Single & Multi-Resolution Active Shape Models' fitting procedure.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''
import cv2
import numpy as np
import math

import classification_utils as cu
import configuration as c
import fitting_function as ff
import fitting_utils as fu
import gaussian_image_piramid as gip
import loader as l
import math_utils as mu
import principal_component_analysis as pca
import procrustes_analysis as pa

from matplotlib import pyplot

MS = None                       # MS contains for each tooth, the tooth model (in the model coordinate frame)
EWS = []                        # EWS contains for each tooth, a (sqrt(Eigenvalues), Eigenvectors) pair (in the model coordinate frame)
fns = None                      # fitting functions for each tooth, for each landmark, for the profile normal through that landmark.
fts = None                      # fitting functions for each tooth, for each landmark, for the profile gradient through that landmark.

offsetY = 497.0                 # The landmarks refer to the non-cropped images, so we need the vertical offset (up->down)
                                # to locate them on the cropped images.
offsetX = 1234.0                # The landmarks refer to the non-cropped images, so we need the horizontal offset (left->right)
                                # to locate them on the cropped images.

k = 4                           # The number of pixels to sample either side for each of the model points along the profile normal
                                # (used for creating the fitting functions)
m = 8                           # The number of pixels to sample either side for each of the model points along the profile normal
                                # (used while iterating)
method='SCD'                    # The method used for preproccesing.

convergence_threshold = 0.002   # The convergence threshold (used while iterating).
tolerable_deviation = 3         # The number of deviations that are tolerable by the models (used for limiting the shape).


max_level = 2                   # Coarsest level of gaussian pyramid (depends on the size of the object in the image)
max_it = 20                     # Maximum number of iterations allowed at each level
pclose = 0.9                    # Desired proportion of points found within m/2 of current position

def multi_resolution_search(img, P, tooth_index, fitting_function=1, show=False):
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
    level = max_level
    pyramids = gip.get_gaussian_pyramids(img, level)
    
    
    # Compute model point positions in image at coarsest level
    P = np.around(np.divide(P, 2**level))
    
    while (level >= 0):
        nb_it += 1
        pxs, pys = mu.extract_coordinates(P)
        for i in range(c.get_nb_landmarks()):
            tx, ty, nx, ny = ff.create_ricos(pyramids[level], i, pxs, pys)
            f_optimal = float("inf")
            
            if (fitting_function==0):
                rn = rt = range(-(m-k), (m-k)+1)
            elif (fitting_function==1):
                rn = range(-(m-k), (m-k)+1)
                rt = [0]
            else:
                rn = [0]
                rt = range(-(m-k), (m-k)+1)
            
            for n in rn:
                for t in rt:
                    x = round(pxs[i] + n * nx + t * tx)
                    y = round(pys[i] + n * ny + t * ty)
                    try:    
                        fn = fns[level][tooth_index][i](ff.normalize_Gi(ff.create_Gi(pyramids[level], k, x, y, nx, ny)))
                        ft = fts[level][tooth_index][i](ff.normalize_Gi(ff.create_Gi(pyramids[level], k, x, y, tx, ty)))
                    except (IndexError): continue
                    f = fu.evaluate_fitting(fn=fn, ft=ft, fitting_function=fitting_function)
                    if f < f_optimal:
                        f_optimal = f
                        cx = x
                        cy = y
            pxs[i] = cx
            pys[i] = cy
            
        P_new = validate(pyramids[level], tooth_index, mu.zip_coordinates(pxs, pys), nb_it, show)
        nb_close_points = nb_closest_points(P, P_new)
        P = P_new
        
        # Repeat unless more than pclose of the points are found close to the current position 
        # or nmax iterations have been applied at this resolution
        print 'Level:' + str(level) + ', Iteration: ' + str(nb_it) + ', Ratio: ' + str((2 * nb_close_points / float(P.shape[0])))

        converged = (2 * nb_close_points / float(P.shape[0]) >= pclose)    
        if (converged or nb_it >= max_it):
            if (level > 0): 
                level -= 1
                nb_it = 0
                P = P * 2
            else:
                break
                
    return P
          
 
def nb_closest_points(P, P_new):
    nb_close_points = 0
    for i in range(P.shape[0] / 2):
        if close_to_current_position(P[(i*2)], P[(i*2+1)], P_new[(i*2)], P_new[(i*2+1)]): 
            nb_close_points += 1
    return nb_close_points
                     
def close_to_current_position(found_x, found_y, current_x, current_y):
    distance = math.sqrt((current_x - found_x) ** 2 + (current_y - found_y) ** 2)
    return distance <= 1.5#(ns / 2)  
                
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
        fu.show_validation(MU, nb_it, PY_before, PY_after)
        fu.show_iteration(np.copy(img), nb_it, P_before, P_after)
        cv2.waitKey(0)
        pyplot.close()  

    return P_after 
    
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
        S = XS[j,:,:]
        M, Y = pa.PA(S)
        MS[j,:] = M
        E, W, MU = pca.pca_percentage(Y)
        EWS.append((np.sqrt(E), W))

    GNS, GTS = ff.create_partial_GS_for_multiple_levels(trainingSamples, XS, MS, (max_level+1), offsetX=fu.offsetX, offsetY=fu.offsetY, k=k, method=method)
    fns, fts = ff.create_fitting_functions_for_multiple_levels(GNS, GTS)

################################################################################
# TESTS
################################################################################
def test1():   
    for i in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(i)
        preprocess(trainingSamples)
        
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        
        for j in range(c.get_nb_teeth()):
            fname = c.get_fname_original_landmark(i, (j+1))
            P = fu.original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))
            R = multi_resolution_search(img, P, j)
            fname = str(i) + '-' + str((j+1)) + '.png'
            cv2.imwrite(fname, fu.mark_results(np.copy(img), np.array([P, R])))  

def test1_combined():
    Results = np.zeros((c.get_nb_trainingSamples(), 2*c.get_nb_teeth(), c.get_nb_dim()))
    color_lines = np.array([np.array([0,0,255]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,255,0]), np.array([0,0,255]), np.array([0,255,0])])  
    for i in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(i)
        preprocess(trainingSamples)
        
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        
        for j in range(c.get_nb_teeth()):
            fname = c.get_fname_original_landmark(i, (j+1))
            P = fu.original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))
            Results[(i-1), (2*j), :] = P
            Results[(i-1), (2*j+1), :] = multi_resolution_search(img, P, j)
        
        fname = str(i) + '.png'
        cv2.imwrite(fname, fu.mark_results(np.copy(img), Results[(i-1),:], color_lines))  
          
def test2():     
    for i in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(i)
        preprocess(trainingSamples)
        
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        
        for f in range(2):
            for j in range(c.get_nb_teeth()):
                fname = c.get_fname_original_landmark(i, (j+1))
                P = fu.original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))
                R = multi_resolution_search(img, P, j, fitting_function=f)
                fname = str(i) + '-' + str((j+1)) + '-f' + str(f) + '.png'
                cv2.imwrite(fname, fu.mark_results(np.copy(img), np.array([P, R])))     

def test2_combined():
    Results = np.zeros((c.get_nb_trainingSamples(), 3*c.get_nb_teeth(), c.get_nb_dim()))
    color_lines = np.array([np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),])        
    for i in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(i)
        preprocess(trainingSamples)
        
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        
        for f in range(2):
            for j in range(c.get_nb_teeth()):
                fname = c.get_fname_original_landmark(i, (j+1))
                P = fu.original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))
                if f==0: Results[(i-1), j, :] = P
                Results[(i-1), (f+1)*c.get_nb_teeth()+j, :] = multi_resolution_search(img, P, j, fitting_function=f)
        
        fname = str(i) + 'm.png'
        cv2.imwrite(fname, fu.mark_results(np.copy(img), Results[(i-1),:], color_lines))
            
def test3_combined():
    BS = cu.create_bboxes(method)
    Avg = cu.get_average_size(method)
    
    Results = np.zeros((c.get_nb_trainingSamples(), 3*c.get_nb_teeth(), c.get_nb_dim()))
    color_lines = np.array([np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,0,255]), np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([0,255,0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),np.array([255, 0, 0]),])        
                                     
    for i in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(i)
        preprocess(trainingSamples)
        
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        
        Params = cu.get_average_params(trainingSamples, method)
        
        x_min = BS[(i-1),0]
        x_max = BS[(i-1),1]
        y_min = BS[(i-1),2]
        y_max = BS[(i-1),3]
        ty = y_min + (y_max-y_min) / 2
        for j in range(c.get_nb_teeth()/2):
            if j==0: tx = x_min + Avg[0, 0] / 2.0
            if j==1: tx = x_min + Avg[0, 0] + Avg[1, 0] / 2.0
            if j==2: tx = x_max - Avg[3, 0] - Avg[2, 0] / 2.0
            if j==3: tx = x_max - Avg[3, 0] / 2.0
    
            P = limit(img, mu.full_align(MS[j,:], tx, ty, Params[j,2], Params[j,3]))
            
            fname = c.get_fname_original_landmark(i, (j+1))
            I = fu.original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))
            Results[(i-1), j, :] = I
            Results[(i-1), c.get_nb_teeth()+j, :] = limit(img, multi_resolution_search(img, P, j)) #only limit for i=9: gigantic fail
            Results[(i-1), 2*c.get_nb_teeth()+j, :] = P
            
        x_min = BS[(i-1),4]
        x_max = BS[(i-1),5]
        y_min = BS[(i-1),6]
        y_max = BS[(i-1),7]
        ty = y_min + (y_max-y_min) / 2
        for j in range(c.get_nb_teeth()/2, c.get_nb_teeth()):
            if j==4: tx = x_min + Avg[4, 0] / 2.0
            if j==5: tx = x_min + Avg[4, 0] + Avg[5, 0] / 2.0
            if j==6: tx = x_max - Avg[7, 0] - Avg[6, 0] / 2.0
            if j==7: tx = x_max - Avg[7, 0] / 2.0
            
            P = limit(img, mu.full_align(MS[j,:], tx, ty, Params[j,2], Params[j,3]))
            
            fname = c.get_fname_original_landmark(i, (j+1))
            I = fu.original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))
            Results[(i-1), j, :] = I
            Results[(i-1), c.get_nb_teeth()+j, :] = limit(img, multi_resolution_search(img, P, j)) #only limit for i=9: gigantic fail
            Results[(i-1), 2*c.get_nb_teeth()+j, :] = P
        
        fname = str(i) + 'c.png'
        cv2.imwrite(fname, fu.mark_results(np.copy(img), Results[(i-1),:], color_lines))
        
def limit(img, P):
    pxs, pys = mu.extract_coordinates(P)
    pys[pys < 0] = 0
    pys[pys >= img.shape[0]] = img.shape[0]-1
    pxs[pxs < 0] = 0
    pxs[pxs >= img.shape[1]] = img.shape[1]-1
    P = mu.zip_coordinates(pxs, pys)
    return P

if __name__ == "__main__":
    #test1_combined()
    #test2_combined()  
    test3_combined() 
