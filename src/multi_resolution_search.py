'''
Single-Resolution Active Shape Models' fitting procedure.
Improve the efficiency and robustness of the ASM algorithm 
by implementing it in a multi-resolution framework.
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
import gaussian_image_piramid as gip
import scipy.spatial.distance as dist
import copy

import math


'''
Model Parameters
'''
n = c.get_nb_teeth()            #Number of model points
t = None                        #Number of modes to use
k = 3                           #Number of pixels either side of point to represent in grey-model 

'''
Search Parameters
'''
lmax = gip.lmax                 #Coarsest level of gaussian pyramid (depends on the size of the object in the image)
ns = 2                          #Number of sample points either side of current point
nmax = 5                        #Maximum number of iterations allowed at each level
pclose = 0.9                    #Desired proportion of points found within ns/2 of current position

'''
Other Parameters
'''
models = None
eigen = []
fitting_functions = None
offsetY = 497.0                 #The landmarks refer to the non-cropped images, so we need the vertical offset (up->down)
                                #to locate them on the cropped images.
offsetX = 1234.0                #The landmarks refer to the non-cropped images, so we need the horizontal offset (left->right)
                                #to locate them on the cropped images.
tolerable_deviation = 3         #The number of deviations that are tolerable by the models (used for limiting the shape).
method='SCD'                    #The method used for preproccesing.


def multi_resolution_search(nr_test_sample, model_points, nr_tooth):
    
    l = lmax
    image = cv2.imread(c.get_fname_pyramids(nr_test_sample, l))
    nb_iterations = 0
    
    
    for i in range(80):
        model_points[i] *= 0.6
        if i % 2 == 0:
            model_points[i] += 55 
        else:
            model_points[i] += 160
    
    
    #Compute model point positions in image at coarsest level
    if not lmax == 0:
        for i in range(model_points.shape[0] / 2):
            model_points[i * 2] = round(model_points[i * 2] / np.power(2, l))
            model_points[i * 2 + 1] = round(model_points[i * 2 + 1] / np.power(2, l))
    
    
    while (l >= 0):

        show_iteration(np.copy(image), 0, model_points, model_points)
        cv2.waitKey(0)
        
        #Search at ns points on profile either side of each current point
        x_coords, y_coords = mu.extract_coordinates(model_points)

        for i in range(n): #For each model point
            otherx, othery, dx, dy = ff.create_ricos(image, i, x_coords, y_coords)
            
            best_fit = float("inf")

            for j in range(-ns, ns + 1):
            
                    x_coord = round(x_coords[i] + j * dx)
                    y_coord = round(y_coords[i] + j * dy)
                    fit = fitting_functions[l][nr_tooth][i](ff.normalize_Gi(ff.create_Gi(image, k, x_coord, y_coord, dy, dx)))
                    if fit < best_fit:
                        best_fit = fit
                        best_x_coord = x_coord
                        best_y_coord = y_coord
                                                   
            x_coords[i] = best_x_coord
            y_coords[i] = best_y_coord
            

        
        #Update pose and shape parameters to fit model to new points
        new_model_points = update_parameters(image, nr_tooth, mu.zip_coordinates(x_coords, y_coords), nb_iterations)
        
        nr_close_points = 0 #The number of points that are found close to the current position
        for i in range(n):
            if close_to_current_position(model_points[i * 2], model_points[i * 2 + 1], new_model_points[i * 2], new_model_points[i * 2 + 1]): 
                nr_close_points += 1
        model_points = new_model_points
        
        #Repeat unless more than pclose of the points are found close to the current position 
        #or nmax iterations have been applied at this resolution
        
        print 'Level:' + str(l) + ', Iteration: ' + str(nb_iterations) + ', Ratio: ' + str(nr_close_points / float(n))

        if (nr_close_points / float(n) >= pclose): 
            converged = False
        else: converged = False
        
        nb_iterations += 1     
        if (converged or nb_iterations >= nmax):
            if (l > 0): 
                l = l - 1
                nb_iterations = 0
                image = cv2.imread(c.get_fname_pyramids(nr_test_sample, l))
                #Compute model point positions in image at level l
                for i in range(model_points.shape[0] / 2):
                    model_points[i * 2] *= 2 
                    model_points[i * 2 + 1] *=2
            else: l = -1 #Break
                
    return model_points
          
          
def close_to_current_position(found_x, found_y, current_x, current_y):
    distance = math.sqrt((current_x - found_x) ** 2 + (current_y - found_y) ** 2)
    return distance <= 1#(ns / 2)  
                
def update_parameters(img, nr_tooth, model_points_before, nb_iterations):
    MU = models[nr_tooth]
    E, W = eigen[nr_tooth]
    xm, ym = mu.get_center_of_gravity(model_points_before)
    tx, ty, s, theta = mu.full_align_params(model_points_before, MU)
    PY_before = mu.full_align(model_points_before, tx, ty, s, theta)   
    bs = pca.project(W, PY_before, MU)
    bs = np.maximum(np.minimum(bs, tolerable_deviation*E), -tolerable_deviation*E)
    PY_after = pca.reconstruct(W, bs, MU)
    P_after = mu.full_align(PY_after, xm, ym, 1.0 / s, -theta)
    return P_after

    
def create_fitting_functions(grey_level_models):      
    return [[[get_fitting_function(level, tooth, landmark, grey_level_models) for landmark in range(c.get_nb_landmarks())] for tooth in range(c.get_nb_teeth())] for level in range(gip.lmax+1)] 
    
    
    
    
def get_fitting_function(level, tooth_index, landmark_index, grey_level_models):
    G = np.zeros((grey_level_models.shape[2], grey_level_models.shape[4]))
    
    #Iterate all the training samples and levels
    for i in range(grey_level_models.shape[2]):    
        G[i,:] = grey_level_models[level, tooth_index, i, landmark_index, :]
        
    g_mu = G.mean(axis=0) #Model mean
    
    for i in range(G.shape[0]):
        G[i,:] -= g_mu
        
    #Use the Moore-Penrose pseudo-inverse because C can be singular
    C = np.linalg.pinv((np.dot(G.T, G) / float(G.shape[0])))

    
    def fitting_function(gs):
        '''
        Calculate the Mahalanobis distance for the given sample.
        @param: gs           the new sample
        @return The Mahalanobis distance for the given sample.
        '''
        return dist.mahalanobis(gs, g_mu, C)

    return fitting_function   
   
   
   
def create_partial_grey_level_models(trainingSamples, XS):
    #A sample for each landmark of each training sample for each tooth and for each level
    grey_level_models = np.zeros((gip.lmax+1, c.get_nb_teeth(), len(trainingSamples), c.get_nb_landmarks(), 2*k+1))
    for tooth in range(c.get_nb_teeth()):
        sample_index = 0
        for sample in trainingSamples:
            
            coordinates = original_to_cropped(XS[tooth,sample_index,:])
            
            # model of tooth j from model coordinate frame to image coordinate frame
            xs, ys = mu.extract_coordinates(mu.full_align_with(models[tooth], coordinates))
            
            for level in range(gip.lmax+1): 
                
                if not level == 0:
                    for i in range(coordinates.shape[0] / 2):
                        coordinates[i*2] = round(coordinates[i*2]/2)
                        coordinates[i*2+1] = round(coordinates[i*2+1]/2)
                    xs, ys = mu.extract_coordinates(mu.full_align_with(models[tooth], coordinates))
                 
                fname = c.get_fname_pyramids(sample, level)
                img = cv2.imread(fname)
                    
                grey_level_model = ff.create_G(img, k, xs, ys)[0]
                grey_level_models[level, tooth, sample_index, :] = grey_level_model
            sample_index += 1
                             
    return grey_level_models
    
    


    
def original_to_cropped(coordinates):
    for i in range(coordinates.shape[0] / 2):
        coordinates[(2*i)] -= offsetX
        coordinates[(2*i+1)] -= offsetY
    return coordinates


def show_iteration(img, nb_it, P_before, P_after, color_init=np.array([0,255,255]), color_mid=np.array([255,0,255]), color_end=np.array([255,255,0]), color_line_before=np.array([0,0,255]), color_line_after=np.array([0,255,0])):
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
    cv2.imshow(txt, img)

def preprocess(trainingSamples):

    global models, eigen, fitting_functions
    XS = l.create_partial_XS(trainingSamples)
    models = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    
    for j in range(c.get_nb_teeth()):
        M, Y = pa.PA(XS[j,:,:])
        models[j,:] = M
        eigenvalues, eigenvectors, MU = pca.pca_percentage(Y)
        eigen.append((np.sqrt(eigenvalues), eigenvectors))

    grey_level_models = create_partial_grey_level_models(trainingSamples, XS)
    fitting_functions = create_fitting_functions(grey_level_models)
    
if __name__ == '__main__':

    for i in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(i)
        preprocess(trainingSamples)
        
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        
        for j in range(c.get_nb_teeth()):
            fname = c.get_fname_original_landmark(i, (j+1))
            P = original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))

            
            R = multi_resolution_search(i, copy.copy(P), j)
            

            
        
