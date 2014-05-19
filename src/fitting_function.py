'''
Construction of the fitting functions (for each tooth, for each landmark)
by sampling the along the profile normal to the boundary in the training set
and building a statistical model of the grey-level structure.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import cv2
import numpy as np
import scipy.spatial.distance as dist
import configuration as c
import math
import math_utils as mu

def create_fitting_functions(GS):
    '''
    Creates the fitting function for each tooth, for each landmark.
    @param GS:              the matrix GS wich contains for each tooth, for each of the given training samples,
                            for each landmark, a normalized sample (along the profile normal through that landmark)
    @return The fitting function for each tooth, for each landmark.
    '''          
    fs = [[get_fitting_function(tooth, landmark, GS) for landmark in range(c.get_nb_landmarks())] for tooth in range(c.get_nb_teeth())]        
    return fs  
    
def get_fitting_function(tooth_index, landmark_index, GS):
    '''
    Creates the fitting function for the given tooth index, for the given landmark index.
    @param tooth_index:     the index of the tooth (in GS)
    @param landmark_index:  the index of the landmark (n GS)
    @param GS:              the matrix GS wich contains for each tooth, for each of the given training samples,
                            for each landmark, a normalized sample (along the profile normal through that landmark)
    @return The fitting function for the given tooth index, for the given landmark index.
    '''
    G = np.zeros((GS.shape[1], GS.shape[3]))
    #Iterate all the training samples
    for i in range(GS.shape[1]):
        G[i,:] = GS[tooth_index, i, landmark_index, :]
    
    G -= G.mean(axis=0)[None, :]
    C = (np.dot(G.T, G) / float(G.shape[0]))
    g_mu = G.mean(axis=0)
    
    def fitting_function(g):
        '''
        Calculate the Mahalanobis distance for the given sample
        @param: g           the sample
        @return The Mahalanobis distance for the given sample
        '''
        #Use the Moore-Penrose pseudo-inverse because C can be singular
        return dist.mahalanobis(g, g_mu, np.linalg.pinv(C))

    return fitting_function  
    
def create_partial_GS(trainingSamples, XS, MS, offsetX=0, offsetY=0, k=5, method=''):
    '''
    Creates the matrix GS wich contains for each tooth, for each of the given training samples,
    for each landmark, a normalized sample (along the profile normal through that landmark).
    @param trainingSamples: the number of the training samples (not the test training samples!)
    @param XS:              contains for each tooth, for each training sample, all landmarks (in the image coordinate frame)
    @param MS:              contains for each tooth, the tooth model (in the model coordinate frame)
    @param offsetX:         the offset in x direction (used when working with corpped images and non-cropped landmarks)
    @param offsetY:         the offset in y direction (used when working with corpped images and non-cropped landmarks)
    @param k:               the number of pixels to sample either side for each of the model points along the profile normal
    @param method:          the method used for preproccesing
    @return The matrix GS wich contains for each tooth, for each of the given training samples,
            for each landmark, a normalized sample (along the profile normal through that landmark).
    '''
    gradients = create_gradients(trainingSamples, method)
    GS = np.zeros((c.get_nb_teeth(), len(trainingSamples), c.get_nb_landmarks(), 2*k+1))
    for j in range(c.get_nb_teeth()):
        for i in range(len(trainingSamples)):
            # tooth j model in model coordinate frame to image coordinate frame
            xs, ys = mu.extract_coordinates(mu.full_align_with(MS[j], XS[j,i,:]))
            GS[j,i,:] = create_G(gradients[i,:], k, xs, ys, offsetX, offsetY)
    return GS
    
def create_gradients(trainingSamples, method=''):
    '''
    Creates the gradient image for each of the given preprocced training samples with the given method.
    @param trainingSamples: the number of the training samples
    @param method:          the method used for preproccesing
    @return The gradient image for each of the given preprocced training samples with the given method.
    '''
    index = 0
    for i in trainingSamples:
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        if index == 0: 
            gradients = np.zeros((len(trainingSamples), img.shape[0], img.shape[1], img.shape[2]))
        gradients[index,:] = create_gradient(img)
        index += 1
    return gradients
    
def create_gradient(img):
    '''
    Creates the gradient image for the given image by differentiate in x and y direction.
    @param img:          the image
    @return The gradient image.
    '''
    #When the size of the kernel is 3, the Sobel kernel may produce noticeable inaccuracies
    #(after all, Sobel is only an approximation of the derivative). OpenCV addresses this
    #inaccuracy for kernels of size 3 by using the Scharr function.
    temp = cv2.Scharr(img, ddepth=-1, dx=1, dy=0)
    return cv2.Scharr(temp, ddepth=-1, dx=0, dy=1)
                 
def create_G(img, k, xs, ys, offsetX=0, offsetY=0):
    '''
    Sample along the profile normal k pixels either side for each of the given model
    points (xs[i], ys[i]) in the given image to create the matrix G, which contains
    for each landmark a normalized sample.
    @param img:          the (gradient) image
    @param k:            the number of pixels to sample either side for each of the
                         given model points (xs[i], ys[i]) along the profile normal
    @param i:            the index of the model point
    @param xs:           x positions of the model points in the image
    @param ys:           y positions of the model points in the image
    @param offsetX:      the offset in x direction (used when working with corpped images and non-cropped xs & ys)
    @param offsetY:      the offset in y direction (used when working with corpped images and non-cropped xs & ys)
    @return The matrix G, which contains for each landmark a normalized sample.
    '''
    G = np.zeros((c.get_nb_landmarks(), 2*k+1))
    for i in range(c.get_nb_landmarks()):
        Gi, Coords = create_Gi(img, k, i, xs, ys, offsetX, offsetY)
        G[i,:] = normalize_Gi(Gi)
    return G
    
def normalize_Gi(Gi):
    '''
    Normalizes the given sample Gi by dividing through by the sum of the
    absolute element values.
    @param Gi:           the sample to normalize.
    @return The normalized sample.
    '''
    norm = 0
    for i in range(Gi.shape[0]):
        norm += abs(Gi[i])
    if norm==0: 
        return Gi
    return Gi/norm
    
def create_Gi(img, k, i, xs, ys, offsetX=0, offsetY=0, sx=1, sy=1):
    '''
    Sample along the profile normal k pixels either side of the given model point (xs[i], ys[i])
    in the given image to create a (non-normalized) vector Gi.
    @param img:          the (gradient) image
    @param k:            the number of pixels to sample either side of the given model
                         point (xs[i], ys[i]) along the profile normal
    @param i:            the index of the model point
    @param xs:           x positions of the model points in the image
    @param ys:           y positions of the model points in the image
    @param offsetX:      the offset in x direction (used when working with corpped images and non-cropped xs & ys)
    @param offsetY:      the offset in y direction (used when working with corpped images and non-cropped xs & ys)
    @param sx:           the step to multiply the profile normal x-change in direction with
    @param sy:           the step to multiply the profile normal y-change in direction with
    @return The (non-normalized) vector Gi and a vector containing the coordinates
            of all the sample points used. (First the most distant point when adding
            a positive change, last the most distant point when adding a negative change) 
    '''
    x = xs[i] - offsetX
    y = ys[i] - offsetY
    if (i == 0):
        x_min = xs[-1] - offsetX
        y_min = ys[-1] - offsetY
        x_max = xs[1] - offsetX
        y_max = ys[1] - offsetY
    elif (i == xs.shape[0]-1):
        x_min = xs[(i-1)] - offsetX
        y_min = ys[(i-1)] - offsetY
        x_max = xs[0] - offsetX
        y_max = ys[0] - offsetY
    else:
        x_min = xs[(i-1)] - offsetX
        y_min = ys[(i-1)] - offsetY
        x_max = xs[(i+1)] - offsetX
        y_max = ys[(i+1)] - offsetY
        
    dx = x_max - x_min
    dy = y_max - y_min
    sq = math.sqrt(dx*dx+dy*dy)
    
    #Profile Normal to Boundary
    nx = (- dy / sq) * sx
    ny = (dx / sq) * sy
    
    #We explicitly don't want a normalized vector at this stage
    return create_raw_Gi(img, k, x, y, nx, ny)
    
        
def create_raw_Gi(img, k, x, y, nx, ny):
    '''
    Sample along the profile normal characterized by (nx, ny) k pixels either side
    of the given model point (x, y) in the given image to create a (non-normalized) vector Gi.
    @param img:          the (gradient) image
    @param k:            the number of pixels to sample either side of the given model
                         point along the profile normal characterized by (nx, ny)
    @param x:            x position of the model point in the image
    @param y:            y position of the model point in the image
    @param nx:           profile normal x-change in direction (step/magnitude included)
    @param ny:           profile normal y-change in direction (step/magnitude included)
    @return The (non-normalized) vector Gi and a vector containing the coordinates
            of all the sample points used. (First the most distant point when adding
            a positive change, last the most distant point when adding a negative change) 
    '''
    Gi = np.zeros((2*k+1))         #2k + 1 samples
    Coords = np.zeros(2*(2*k+1))
    
    index = 0
    for i in range(k,0,-1):
        kx = int(x + i * nx)
        ky = int(y + i * ny)
        Gi[index] = img[ky,kx,0]
        Coords[(2*index)] = kx
        Coords[(2*index+1)] = ky
        index += 1
        
    Gi[index] = img[y,x,0] #The model point itself
    Coords[(2*index)] = x
    Coords[(2*index+1)] = y
    index += 1
        
    for i in range(1,k+1):
        kx = int(x - i * nx)
        ky = int(y - i * ny)
        Gi[index] = img[ky,kx,0]
        Coords[(2*index)] = kx
        Coords[(2*index+1)] = ky
        index += 1
    
    #We explicitly don't want a normalized vector at this stage
    return Gi, Coords
