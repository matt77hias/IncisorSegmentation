'''
Gaussian Image Piramid
A gaussian image piramid of an image is formed 
by repeated smoothing and sub-sampling.
Used in the multi-resolution search algorithm.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''
import cv2

def get_gaussian_pyramids(img, level):
    pyramids = [img]
    pyramid = img
    for i in range(level):
        pyramid = cv2.pyrDown(pyramid)
        pyramids.append(pyramid)
    return pyramids
        
def get_gaussian_pyramid_at(img, level):
    pyramid = img
    for i in range(level):
        pyramid = cv2.pyrDown(pyramid)
    return pyramid
        