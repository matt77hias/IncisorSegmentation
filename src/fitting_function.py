import cv2
import numpy as np
import configuration as c


def create_gradient_images(trainingSamples, method=''):
    gradients = np.zeros(len(trainigsamples), 
    for i in trainingSamples:
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        gradient = cv2.Scharr(img, ddepth=-1, dx=1, dy=1)
    

def create_GS(trainingSamples, XS, offsetX, offsetY, k, method=''):
    for j in range(c.get_nb_teeth()):
        for i in trainingSamples:
            fname = c.get_fname_vis_pre(i, method)
            img = cv2.imread(fname)
            gradient = cv2.Scharr(img, ddepth=-1, dx=1, dy=1)
            trainingsample = XS[j,i,:]
            
def create_Gi
            
            
            

    


