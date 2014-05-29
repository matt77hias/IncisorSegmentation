'''
Construction of the fitting functions (for each tooth, for each landmark)
by sampling along the profile normal and tangent to the boundary in the
training set and building a statistical model of the grey-level structure.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import cv2
import numpy as np
import scipy.spatial.distance as dist
import configuration as c
import gaussian_image_piramid as gip
import math
import math_utils as mu

def create_fitting_functions_for_multiple_levels(L_GNS, L_GTS):
    '''
    Creates the fitting function for each level, for each tooth, for each landmark.
    @param GNS:              the matrix L_GNS which contains for each level, for each tooth, for each of the given training samples,
                             for each landmark, a normalized sample (along the profile normal through that landmark)
    @param GTS:              the matrix L_GTS which contains for each level, for each tooth, for each of the given training samples,
                             for each landmark, a normalized sample (along the profile tangent through that landmark)
    @return The fitting functions for each level, for each tooth, for each landmark.
    '''          
    l_fns = [[[get_fitting_function(tooth, landmark, L_GNS[level,:]) for landmark in range(c.get_nb_landmarks())] for tooth in range(c.get_nb_teeth())] for level in range(L_GNS.shape[0])]
    l_fts = [[[get_fitting_function(tooth, landmark, L_GTS[level,:]) for landmark in range(c.get_nb_landmarks())] for tooth in range(c.get_nb_teeth())] for level in range(L_GTS.shape[0])]   
    return l_fns, l_fts 

def create_fitting_functions(GNS, GTS):
    '''
    Creates the fitting function for each tooth, for each landmark.
    @param GNS:              the matrix GNS which contains for each tooth, for each of the given training samples,
                             for each landmark, a normalized sample (along the profile normal through that landmark)
    @param GTS:              the matrix GTS which contains for each tooth, for each of the given training samples,
                             for each landmark, a normalized sample (along the profile tangent through that landmark)
    @return The fitting functions for each tooth, for each landmark.
    '''          
    fns = [[get_fitting_function(tooth, landmark, GNS) for landmark in range(c.get_nb_landmarks())] for tooth in range(c.get_nb_teeth())]
    fts = [[get_fitting_function(tooth, landmark, GTS) for landmark in range(c.get_nb_landmarks())] for tooth in range(c.get_nb_teeth())]        
    return fns, fts  
    
def get_fitting_function(tooth_index, landmark_index, GS):
    '''
    Creates the fitting function for the given tooth index, for the given landmark index.
    @param tooth_index:     the index of the tooth (in GS)
    @param landmark_index:  the index of the landmark (in GS)
    @param GS:              the matrix GS which contains for each tooth, for each of the given training samples,
                            for each landmark, a normalized sample (along the profile normal/tangent through that landmark)
    @return The fitting function for the given tooth index, for the given landmark index.
    '''
    G = np.zeros((GS.shape[1], GS.shape[3]))
    #Iterate all the training samples
    for i in range(GS.shape[1]):
        G[i,:] = GS[tooth_index, i, landmark_index, :]
    
    g_mu = G.mean(axis=0)
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
        #return np.dot(np.transpose(gs - g_mu), np.dot(np.linalg.pinv(C), (gs - g_mu)))
        return dist.mahalanobis(gs, g_mu, C)

    return fitting_function     

def create_partial_GS_for_multiple_levels(trainingSamples, XS, MS, nb_levels=1, offsetX=0, offsetY=0, k=5, method=''):
    '''
    Creates the matrix L_GNS which contains for each level, for each tooth, for each of the given training samples,
    for each landmark, a normalized sample (along the profile normal through the landmarks).
    Creates the matrix L_GTS which contains for each tooth, for each of the given training samples,
    for each landmark, a normalized sample (along the profile tangent through the landmarks).
    @param trainingSamples: the number of the training samples (not the test training samples!)
    @param XS:              contains for each tooth, for each training sample, all landmarks (in the image coordinate frame)
    @param MS:              contains for each tooth, the tooth model (in the model coordinate frame)
    @param nb_levels:           the number of levels
    @param offsetX:         the possible offset in x direction (used when working with cropped images and non-cropped landmarks)
    @param offsetY:         the possible offset in y direction (used when working with cropped images and non-cropped landmarks)
    @param k:               the number of pixels to sample either side for each of the model points along the profile normal
    @param method:          the method used for preprocessing
    @return The matrix L_GNS which contains for each level, for each tooth, for each of the given training samples,
            for each landmark, a normalized sample (along the profile normal through that landmark).
            The matrix L_GTS which contains for each level, for each tooth, for each of the given training samples,
            for each landmark, a normalized sample (along the profile tangent through that landmark).
    '''
    L_GNS = np.zeros((nb_levels, c.get_nb_teeth(), len(trainingSamples), c.get_nb_landmarks(), 2*k+1))
    L_GTS = np.zeros((nb_levels, c.get_nb_teeth(), len(trainingSamples), c.get_nb_landmarks(), 2*k+1))
    for i in range(nb_levels):
        GNS, GTS = create_partial_GS(trainingSamples, np.around(np.divide(XS, 2**i)), MS, i, offsetX=round(float(offsetX)/2**i), offsetY=round(float(offsetY)/2**i), k=k, method=method)
        L_GNS[i,:] = GNS
        L_GTS[i,:] = GTS
    return L_GNS, L_GTS  

def create_partial_GS(trainingSamples, XS, MS, level=0, offsetX=0, offsetY=0, k=5, method=''):
    '''
    Creates the matrix GNS which contains for each tooth, for each of the given training samples,
    for each landmark, a normalized sample (along the profile normal through the landmarks).
    Creates the matrix GTS which contains for each tooth, for each of the given training samples,
    for each landmark, a normalized sample (along the profile tangent through the landmarks).
    @param trainingSamples: the number of the training samples (not the test training samples!)
    @param XS:              contains for each tooth, for each training sample, all landmarks (in the image coordinate frame)
    @param MS:              contains for each tooth, the tooth model (in the model coordinate frame)
    @param level:           the current level
    @param offsetX:         the possible offset in x direction (used when working with cropped images and non-cropped landmarks)
    @param offsetY:         the possible offset in y direction (used when working with cropped images and non-cropped landmarks)
    @param k:               the number of pixels to sample either side for each of the model points along the profile normal
    @param method:          the method used for preprocessing
    @return The matrix GNS which contains for each tooth, for each of the given training samples,
            for each landmark, a normalized sample (along the profile normal through that landmark).
            The matrix GTS which contains for each tooth, for each of the given training samples,
            for each landmark, a normalized sample (along the profile tangent through that landmark).
    '''
    GNS = np.zeros((c.get_nb_teeth(), len(trainingSamples), c.get_nb_landmarks(), 2*k+1))
    GTS = np.zeros((c.get_nb_teeth(), len(trainingSamples), c.get_nb_landmarks(), 2*k+1))
    for j in range(c.get_nb_teeth()):
        index = 0
        for i in trainingSamples:
            # model of tooth j from model coordinate frame to image coordinate frame
            xs, ys = mu.extract_coordinates(mu.full_align_with(MS[j], XS[j,index,:]))
            fname = c.get_fname_vis_pre(i, method)
            img = cv2.imread(fname)
            pyramid = gip.get_gaussian_pyramid_at(img, level)
            GN, GT = create_G(pyramid, k, xs, ys, offsetX, offsetY)
            GNS[j,index,:] = GN
            GTS[j,index,:] = GT
            index += 1
    return GNS, GTS
                 
def create_G(img, k, xs, ys, offsetX=0, offsetY=0):
    '''
    Sample along the profile normal and profile tangent k pixels either side for
    each of the given model points (xs[i], ys[i]) in the given image to create
    the matrix G, which contains for each landmark a normalized sample.
    @param img:          the image
    @param k:            the number of pixels to sample either side for each of the
                         given model points (xs[i], ys[i]) along the profile normal
    @param i:            the index of the model point
    @param xs:           x positions of the model points in the image
    @param ys:           y positions of the model points in the image
    @param offsetX:      the possible offset in x direction 
                         (used when working with cropped images and non-cropped xs & ys)
    @param offsetY:      the possible offset in y direction
                         (used when working with cropped images and non-cropped xs & ys)
    @return The matrix GN, which contains for each landmark a normalized sample 
            (sampled along the profile normal through the landmarks).
            The matrix GT, which contains for each landmark a normalized sample 
            (sampled along the profile tangent through the landmarks).
    '''
    GN = np.zeros((c.get_nb_landmarks(), 2*k+1))
    GT = np.zeros((c.get_nb_landmarks(), 2*k+1))
    for i in range(c.get_nb_landmarks()):
        x = xs[i] - offsetX
        y = ys[i] - offsetY
        tx, ty, nx, ny = create_ricos(img, i, xs, ys)
        GN[i,:] = normalize_Gi(create_Gi(img, k, x, y, nx, ny))
        GT[i,:] = normalize_Gi(create_Gi(img, k, x, y, tx, ty))
    return GN, GT
    
def normalize_Gi(Gi):
    '''
    Normalizes the given sample Gi by dividing through by the sum of the
    absolute element values.
    @param Gi:           the sample to normalize
    @return The normalized sample.
    '''
    norm = 0
    for j in range(Gi.shape[0]):
        norm += abs(Gi[j])
    if norm==0: 
        return Gi
    return Gi/norm
    
def create_ricos(img, i, xs, ys):
    '''
    Returns the rico of the profile tangent and the rico of the profile normal
    through the model point at the given index.
    @param img:          the image
    @param i:            the index of the model point
    @param xs:           x positions of the model points in the image
    @param ys:           y positions of the model points in the image
    @return The rico of the profile tangent and the rico of the profile normal
            through the model point at the given index.
    '''
    if (i == 0):
        x_min = xs[-1]
        y_min = ys[-1]
        x_max = xs[1]
        y_max = ys[1]
    elif (i == xs.shape[0]-1):
        x_min = xs[(i-1)]
        y_min = ys[(i-1)]
        x_max = xs[0]
        y_max = ys[0]
    else:
        x_min = xs[(i-1)]
        y_min = ys[(i-1)]
        x_max = xs[(i+1)]
        y_max = ys[(i+1)]
        
    dx = x_max - x_min
    dy = y_max - y_min
    sq = math.sqrt(dx*dx+dy*dy)
    
    #Profile Tangent to Boundary
    tx = (dx / sq)
    ty = (dy / sq)
    #Profile Normal to Boundary
    nx = - ty
    ny = tx
    
    return tx, ty, nx, ny
        
def create_Gi(img, k, x, y, dx, dy):
    '''
    Sample along the profile line characterized by (dx, dy) k pixels either side
    of the given model point (x, y) in the given image to create a (non-normalized) vector Gi.
    @param img:          the image
    @param k:            the number of pixels to sample either side of the given model
                         point along the profile normal characterized by (nx, ny)
    @param x:            x position of the model point in the image
    @param y:            y position of the model point in the image
    @param dx:           profile line x-change in direction (step/magnitude included)
    @param dy:           profile line y-change in direction (step/magnitude included)
    @return The (non-normalized) vector Gi. (First the most distant point when adding
            a positive change, last the most distant point when adding a negative change) 
    '''
    Gi = np.zeros((2*k+2))    
    index = 0
    for i in range(k,-(k+2),-1):
        kx = round(x + i * dx)
        ky = round(y + i * dy) 
        Gi[index] = img[ky,kx,0]
        index += 1
    
    Gi = (Gi[1:] - Gi[:-1])
    
    #We explicitly don't want a normalized vector at this stage
    return Gi
