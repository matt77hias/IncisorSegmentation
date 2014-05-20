'''
Single-Resolution Active Shape Models' fitting procedure.
Improve the efficiency and robustness of the ASM algorithm 
by implementing it in a multi-resolution framework.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import cv2
import configuration as c

lmax = 3 #Coarsest level of gaussian pyramid (depends on the size of the object in the image)
ns = 2 #Number of sample points either side of current point
nmax = 5 #Maximum number of iterations allowed at each level
pclose = 0.9 #Desired proportion of points found within ns/2 of current position


def multi_resolution_search(nr_testSample):
    l = lmax
    image = cv2.imread(c.get_fname_pyramids(nr_testSample, l))
    nb_iterations = 0
    while (l >= 0):
        
        #Compute model point positions in image at level l
        '''
        TODO
        '''
        
        #Search at ns points on profile either side each current points
        '''
        TODO
        '''
        
        #Update pose and shape parameters to fit model to new points
        '''
        TODO
        '''
        
        #Repeat unless more than pclose of the points are found close to the current position 
        #or nmax iterations have been applied at this resolution
        if ():
            '''
            TODO
            '''
            converged = True
        else:
            converged = False
        if (converged or nb_iterations >= nmax):
            if (l > 0): 
                l = l - 1
                nb_iterations = 0
                image = cv2.imread(c.get_fname_pyramids(nr_testSample, l))
            

if __name__ == '__main__':
    nr_testSample = 15
    multi_resolution_search(nr_testSample)
