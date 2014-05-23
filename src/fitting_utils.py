'''
Some fitting utilities.
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import configuration as c
import numpy as np
import cv2
import math_utils as mu

from matplotlib import pyplot

offsetY = 497.0                 #The landmarks refer to the non-cropped images, so we need the vertical offset (up->down)
                                #to locate them on the cropped images.
offsetX = 1234.0                #The landmarks refer to the non-cropped images, so we need the horizontal offset (left->right)
                                #to locate them on the cropped images.

def evaluate_fitting(fn=0, ft=0, fitting_function=0):
    ''''
    Evaluates the fitting function.
    @param fn:                  the fitting function evaluated along the profile normal
    @param ft:                  the fitting function evaluated along the profile gradient
    @param fitting_function:    the fitting function used
                                * 0: fitting function along profile normal through landmark +
                                     fitting function along profile gradient through landmark
                                * 1: fitting function along profile normal through landmark
                                * 2: fitting function along profile gradient through landmark
    @return The evaluated fitting function.
    '''
    if (fitting_function==0):
        return fn**2 + ft**2
    elif (fitting_function==1):
        return fn
    elif (fitting_function==2):
        return ft
    else:
        return 0

def original_to_cropped(P):
    '''
    Crops the given points. Used when working with non-cropped initial target points.
    The whole fitting procdure itself doesn't work with offsets at all.
    @pre    The coordinates are stored as successive xi, yi, xj, yj, ...
    @param  P:   the points to crop
    @return The cropped version of P.
    '''
    for i in range(P.shape[0] / 2):
        P[(2*i)] -= offsetX
        P[(2*i+1)] -= offsetY
    return P
    
def show_validation(M, nb_it, PY_before, PY_after):
    '''
    Plots the landmarks corresponding to the mean shape in the model coordinate frame
    and the landmarks corresponding to the current points in the model coordinate frame
    @param M:               the model for the tooth
    @param nb_it:           the number of this iteration
    @param PY_before:       the current points for the target tooth before validation
                            in the model coordinate frame
    @param PY_after:        the current points for the target tooth after validation
                            in the model coordinate frame
    '''
    mxs, mys = mu.extract_coordinates(M)
    mxs = mu.make_circular(mxs)
    mys = mu.make_circular(mys)
    rxs, rys = mu.extract_coordinates(PY_before)
    rxs = mu.make_circular(rxs)
    rys = mu.make_circular(rys)
    gxs, gys = mu.extract_coordinates(PY_after)
    gxs = mu.make_circular(gxs)
    gys = mu.make_circular(gys)
    
    pyplot.figure(1)
    # x coordinates , y coordinates
    pyplot.plot(mxs, mys, '-+b')
    pyplot.plot(rxs, rys, '-+r')
    pyplot.plot(gxs, gys, '-+g')
    txt = 'Model Coordinate Frame - Iteration: ' + str(nb_it)
    pyplot.title(txt)
    pyplot.xlabel('x\'')
    pyplot.ylabel('y\'')
    pyplot.gca().invert_yaxis()
    pyplot.axis('equal')
    pyplot.show()
    
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
    #cv2.imshow(txt, img)
    return img