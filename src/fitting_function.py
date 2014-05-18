import cv2
import numpy as np
import scipy.spatial.distance as dist
import configuration as c
import math
import math_utils as mu

def create_fitting_functions(GS):          
    fs = [[get_fitting_function(tooth, landmark, GS) for landmark in range(c.get_nb_landmarks())] for tooth in range(c.get_nb_teeth())]        
    return fs  
    
def get_fitting_function(nr_tooth, nr_landmark, GS):
    G = np.zeros((GS.shape[1], GS.shape[3]))
    for i in range(GS.shape[1]):
        G[i,:] = GS[nr_tooth, i, nr_landmark, :]
    
    G -= G.mean(axis=0)[None, :]
    C = (np.dot(G.T, G) / float(G.shape[0]))
    g_mu = G.mean(axis=0)
    
    def fitting_function(g):
        return dist.mahalanobis(g, g_mu, np.linalg.pinv(C))

    return fitting_function  
    
def create_partial_GS(trainingSamples, XS, MS, offsetX=0, offsetY=0, k=5, method=''):
    gradients = create_gradients(trainingSamples, method)
    GS = np.zeros((c.get_nb_teeth(), len(trainingSamples), c.get_nb_landmarks(), 2*k+1))
    for j in range(c.get_nb_teeth()):
        for i in range(len(trainingSamples)):
            # tooth j model in model coordinate frame to image coordinate frame
            xs, ys = mu.extract_coordinates(mu.full_align_with(MS[j], XS[j,i,:]))
            GS[j,i,:] = create_G(gradients[i,:], k, xs, ys, offsetX, offsetY)
    return GS
    
def create_gradients(trainingSamples, method=''):
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
    temp = cv2.Scharr(img, ddepth=-1, dx=1, dy=0)
    return cv2.Scharr(temp, ddepth=-1, dx=0, dy=1)
                 
def create_G(img, k, xs, ys, offsetX=0, offsetY=0):
    G = np.zeros((c.get_nb_landmarks(), 2*k+1))
    for i in range(c.get_nb_landmarks()):
        Gi, Coords = create_Gi(img, k, i, xs, ys, offsetX, offsetY)
        G[i,:] = mu.normalize_vector(Gi)
    return G
    
def create_Gi(img, k, i, xs, ys, offsetX=0, offsetY=0, sx=1, sy=1):
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
    Sample along a profile k pixels either side of the model point in the training image
    to create a vector Gi.
    @param img:             the training sample
    @param k:               the number of pixels we sample either side of the model point along a profile
    @param x:               x position of the model point in the image
    @param y:               y position of the model point in the image
    @param nx:              profile normal x
    @param ny:              profile normal y
    @return 
    '''
    
    Gi = np.zeros((2*k+1)) #2k + 1 samples
    Coords = np.zeros(2*(2*k+1))
    
    index = 0
    for i in range(1,k+1):
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
