import cv2
import numpy as np
import loader as l
import configuration as c
import procrustes_analysis as pa
import fitting_utils as fu
import math_utils as mu

MS = None
IS = None
method = 'SCD'

def preprocess(trainingSamples):
    global MS, IS
    XS = l.create_partial_XS(trainingSamples)
    MS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    IS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    
    for j in range(c.get_nb_teeth()):
        S = XS[j,:,:]
        M, Y = pa.PA(S)
        MS[j,:] = M
        
        mtx = mty = ms = mtheta = 0
        n = S.shape[0]
        for i in range(n):
            tx, ty, s, theta = mu.full_align_params(M, fu.original_to_cropped(S[i,:]))
            mtx += tx
            mty += ty
            ms += s
            mtheta += theta
        n = float(n)
        mtx /= n
        mty /= n
        ms /= n
        mtheta /= n
        IS[j,:] = mu.full_align(M, mtx, mty, ms, mtheta)      
    
def test():
    for i in c.get_trainingSamples_range():
        trainingSamples = c.get_trainingSamples_range()
        trainingSamples.remove(i)
        preprocess(trainingSamples)
        
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        
        for j in range(c.get_nb_teeth()):
            fname = c.get_fname_original_landmark(i, (j+1))
            P = fu.original_to_cropped(np.fromfile(fname, dtype=float, count=-1, sep=' '))
            fname = str(i) + '-' + str((j+1)) + '.png'
            cv2.imwrite(fname, fu.show_iteration(np.copy(img), 10000, P, IS[j,:]))

            
if __name__ == "__main__":
    test() 