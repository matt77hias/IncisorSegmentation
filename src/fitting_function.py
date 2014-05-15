import cv2
import numpy as np
import scipy as sp
import configuration as c
import loader as l
import math
import math_utils as mu
import procrustes_analysis as pa

def create_fitting_functions(GS):          
    fs = [[get_fitting_function(tooth, landmark, GS) for landmark in range(c.get_nb_landmarks())] for tooth in range(c.get_nb_teeth())]        
    return fs  
    
def get_fitting_function(nr_tooth, nr_landmark, GS):
    G = np.zeros((GS.shape[1], GS.shape[3]))
    for i in range(GS.shape[1]):
        G[i,:] = GS[nr_tooth, i, nr_landmark, :]
    
    G -= G.mean(axis=0)[None, :]
    C = (np.dot(G, G.T) / float(G.shape[0]))
    #Cholesky decomposition uses half of the operations as LU
    #and is numerically more stable.
    #L = np.linalg.cholesky(C).T
    
    def fitting_function(g):
        x = g - G.mean(axis=0) 
        #z = np.linalg.solve(L, x)
        #Mahalanobis Distance 
        #MD = z.T*z
        #return math.sqrt(MD)
        return sp.spatial.distance.mahalanobis(x.T, np.linalg.inv(C), x)
    
    return fitting_function  
    
def create_partial_GS(trainingSamples, XS, MS, offsetX=0, offsetY=0, k=5, method=''):
    gradients = create_gradient_images(trainingSamples, method)
    GS = np.zeros((c.get_nb_teeth(), len(trainingSamples), c.get_nb_landmarks(), 2*k+1))
    for j in range(c.get_nb_teeth()):
        index = 0
        for i in trainingSamples:
            # tooth j model in model coordinate frame to image coordinate frame
            X = XS[j,index,:]
            M = mu.align_with(mu.center_on(MS[j], X), X)
            xs, ys = mu.extract_coordinates(M)
            
            GS[j,index,:] = create_G(gradients[j,:], xs, ys, offsetX, offsetY, k)
            index += 1
    return GS
    
def create_gradient_images(trainingSamples, method=''):
    index = 0
    for i in trainingSamples:
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        if index==0: 
            gradients = np.zeros((len(trainingSamples), img.shape[0], img.shape[1], img.shape[2]))
        temp = cv2.Scharr(img, ddepth=-1, dx=1, dy=0)
        gradients[index,:] = cv2.Scharr(temp, ddepth=-1, dx=0, dy=1)
        index += 1
    return gradients
                 
def create_G(img, xs, ys, offsetX, offsetY, k):
    G = np.zeros((c.get_nb_landmarks(), 2*k+1))
    for i in range(c.get_nb_landmarks()):
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
        nx = (- dy / sq)
        ny = (dx / sq)
        
        G[i,:] = create_Gi(img, k, x, y, nx, ny)
    return G
        
def create_Gi(img, k, x, y, nx, ny, sx=1, sy=1):
    nx *= sx
    ny *= sy
    
    Gi = np.zeros((2*k+1))
    Gi[0] = img[y,x,0]
    index = 1
    for i in range(1,k+1):
        kx = int(x + i * nx)
        ky = int(y + i * ny)
        Gi[index] = img[ky,kx,0]
        index += 1
    for i in range(1,k+1):
        kx = int(x - i * nx)
        ky = int(y - i * ny)
        Gi[index] = img[ky,kx,0]
        index += 1
    return mu.normalize_vector(Gi)
    
def preprocess(trainingSamples):
    offsetY = 497.0
    offsetX = 1234.0
    XS = l.create_partial_XS(trainingSamples)
    YS = np.zeros((c.get_nb_teeth(), len(trainingSamples), c.get_nb_dim()))
    MS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    for j in range(c.get_nb_teeth()):
        M, Y = pa.PA(l.create_full_X(nr_tooth=1))
        MS[j,:] = M
        YS[j,:] = Y
    
    GS = create_partial_GS(trainingSamples, XS, MS, offsetX=offsetX, offsetY=offsetY, k=5, method='SCD')
    fs = create_fitting_functions(GS)
    return XS, YS, MS, fs
    

if __name__ == "__main__":
    preprocess(c.get_trainingSamples_range())
    #E, W, MU = pca.pca_percentage(Y)