'''
Gaussian Image Piramid
A gaussian image piramid of an image is formed 
by repeated smoothing and sub-sampling.
Used in the multi-resolution search algorithm.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''
import cv2
import configuration as c

lmax = 2                        #Coarsest level of gaussian pyramid (depends on the size of the object in the image)
method='SCD'                    #The method used for preproccesing.

def form_pyramids():
    '''
    Form gaussian image pyramids of all the training and test samples.
    '''
    
    for i in c.get_trainingSamples_range():
        test_image = cv2.imread(c.get_fname_vis_pre(i, method))
        for j in range(lmax+1):
            if j == 0:
                fname = c.get_fname_pyramids(i, j) 
                cv2.imwrite(fname, test_image)
            else:
                test_image = cv2.pyrDown(test_image)
                fname = c.get_fname_pyramids(i, j) 
                cv2.imwrite(fname, test_image)
    '''
    for i in c.get_testSamples_range():
        test_image = cv2.imread(c.get_fname_test_radiograph(i))
        for j in range(lmax+1):
            if j == 0:
                fname = c.get_fname_pyramids(i, j) 
                cv2.imwrite(fname, test_image)
            else:
                test_image = cv2.pyrDown(test_image)
                fname = c.get_fname_pyramids(i, j) 
                cv2.imwrite(fname, test_image)
    '''

if __name__ == '__main__':
    form_pyramids()