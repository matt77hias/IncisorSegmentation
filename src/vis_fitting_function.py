'''
Contains some visualization functions for displaying the intermediate results
while constructing the fitting functions (for each tooth, for each landmark).
@author     Matthias Moulin & Milan Samyn
@version    1.0
'''

import numpy as np
import cv2
import configuration as c
import loader as l
import math_utils as mu
import procrustes_analysis as pa
import fitting_function as ff

XS = None           #XS contains for each tooth, for each training sample, all landmarks (in the image coordinate frame)
MS = None           #MS contains for each tooth, the tooth model (in the model coordinate frame)
offsetY = 497.0     #The landmarks refer to the non-cropped images, so we need the vertical offset (up->down) to locate them on the cropped images.
offsetX = 1234.0    #The landmarks refer to the non-cropped images, so we need the horizontal offset (left->right) to locate them on the cropped images.
    
def create_all_landmarks_images():
    '''
    Stores all the preprocessed images corresponding to all methods used with the landmarks
    of the training samples marked.
    '''
    create_landmarks_images(method='SC')
    create_landmarks_images(method='SCD')
    create_landmarks_images(method='EH')
    create_landmarks_images(method='EHD')    
                
def create_landmarks_images(color_init=np.array([0,255,255]), color_mid=np.array([255,0,255]), color_end=np.array([255,255,0]), color_line=np.array([0,0,255]), method=''):
    '''
    Stores all the preprocessed images corresponding to the given method with the landmarks
    of the training samples marked.
    @param color_init:  the BGR color for the first landmark 
    @param color_mid:   the BGR color for all landmarks except the first and last landmark
    @param color_end:   the BGR color for the last landmark
    @param color_line:  the BGR color for the line between two consecutive landmarks
    @param method:      the method used for preproccesing
    '''
    for i in c.get_trainingSamples_range():
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        for j in range(c.get_nb_teeth()):
            xs, ys = mu.extract_coordinates(XS[j,(i-1),:])
            
            for k in range(c.get_nb_landmarks()):
                x = int(xs[k] - offsetX)
                y = int(ys[k] - offsetY)
                if (k == c.get_nb_landmarks()-1):
                    x_succ = int(xs[0] - offsetX)
                    y_succ = int(ys[0] - offsetY)
                else:
                    x_succ = int(xs[(k+1)] - offsetX)
                    y_succ = int(ys[(k+1)] - offsetY)
                cv2.line(img, (x,y), (x_succ,y_succ), color_line)
          
            for k in range(c.get_nb_landmarks()):
                x = int(xs[k] - offsetX)
                y = int(ys[k] - offsetY)
                if (k == 0):
                    img[y,x] = color_init
                elif (k == c.get_nb_landmarks()-1):
                    img[y,x] = color_end
                else:
                    img[y,x] = color_mid
                
            fname = c.get_fname_vis_ff_landmarks(i, method)
            cv2.imwrite(fname, img)
            
def create_all_landmarks_and_models_images():
    '''
    Stores all the preprocessed images corresponding to all the method used with the landmarks
    of the training samples and models (transformed to the image coordinate system) marked.
    '''
    create_landmarks_and_models_images(method='SC')
    create_landmarks_and_models_images(method='SCD')
    create_landmarks_and_models_images(method='EH')
    create_landmarks_and_models_images(method='EHD')
    
def create_landmarks_and_models_images(color_init=np.array([0,255,255]), color_mid=np.array([255,0,255]), color_end=np.array([255,255,0]), color_line=np.array([0,0,255]), color_model_line=np.array([255,0,0]), method=''):
    '''
    Stores all the preprocessed images corresponding to the given method with the landmarks
    of the training samples and models (transformed to the image coordinate system) marked.
    @param color_init:  the BGR color for the first landmark 
    @param color_mid:   the BGR color for all landmarks except the first and last landmark
    @param color_end:   the BGR color for the last landmark
    @param color_line:  the BGR color for the line between two consecutive landmarks of the training samples
    @param color_model_line:    the BGR color for the line between two consecutive landmarks of the models
    @param method:      the method used for preproccesing
    '''
    for i in c.get_trainingSamples_range():
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        for j in range(c.get_nb_teeth()):
            xs, ys = mu.extract_coordinates(XS[j,(i-1),:])
            mxs, mys = mu.extract_coordinates(mu.full_align_with(MS[j], XS[j,(i-1),:]))
            
            for k in range(c.get_nb_landmarks()):
                x = int(xs[k] - offsetX)
                y = int(ys[k] - offsetY)
                mx = int(mxs[k] - offsetX)
                my = int(mys[k] - offsetY)
                if (k == c.get_nb_landmarks()-1):
                    x_succ = int(xs[0] - offsetX)
                    y_succ = int(ys[0] - offsetY)
                    mx_succ = int(mxs[0] - offsetX)
                    my_succ = int(mys[0] - offsetY)
                else:
                    x_succ = int(xs[(k+1)] - offsetX)
                    y_succ = int(ys[(k+1)] - offsetY)
                    mx_succ = int(mxs[(k+1)] - offsetX)
                    my_succ = int(mys[(k+1)] - offsetY)
                cv2.line(img, (x,y), (x_succ,y_succ), color_line)
                cv2.line(img, (mx,my), (mx_succ,my_succ), color_model_line)
          
            for k in range(c.get_nb_landmarks()):
                x = int(xs[k] - offsetX)
                y = int(ys[k] - offsetY)
                mx = int(mxs[k] - offsetX)
                my = int(mys[k] - offsetY)
                if (k == 0):
                    img[y,x] = color_init
                    img[my,mx] = color_init
                elif (k == c.get_nb_landmarks()-1):
                    img[y,x] = color_end
                    img[my,mx] = color_end
                else:
                    img[y,x] = color_mid
                    img[my,mx] = color_mid
                
            fname = c.get_fname_vis_ff_landmarks_and_models(i, method)
            cv2.imwrite(fname, img) 
                        
def create_all_models_images():
    '''
    Stores all the preprocessed images corresponding to all the method used with the landmarks
    of the models (transformed to the image coordinate system) marked.
    ''' 
    create_models_images(method='SC')
    create_models_images(method='SCD')
    create_models_images(method='EH')
    create_models_images(method='EHD') 
    
def create_models_images(color_init=np.array([0,255,255]), color_mid=np.array([255,0,255]), color_end=np.array([255,255,0]), color_line=np.array([255,0,0]), method=''):
    '''
    Stores all the preprocessed images corresponding to the given method with the landmarks
    of the models (transformed to the image coordinate system) marked.
    @param color_init:  the BGR color for the first landmark
    @param color_mid:   the BGR color for all landmarks except the first and last landmark
    @param color_end:   the BGR color for the last landmark
    @param color_line:  the BGR color for the line between two consecutive landmarks
    @param method:      the method used for preproccesing
    '''
    for i in c.get_trainingSamples_range():
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        for j in range(c.get_nb_teeth()):
            xs, ys = mu.extract_coordinates(mu.full_align_with(MS[j], XS[j,(i-1),:]))
            
            for k in range(c.get_nb_landmarks()):
                x = int(xs[k] - offsetX)
                y = int(ys[k] - offsetY)
                if (k == c.get_nb_landmarks()-1):
                    x_succ = int(xs[0] - offsetX)
                    y_succ = int(ys[0] - offsetY)
                else:
                    x_succ = int(xs[(k+1)] - offsetX)
                    y_succ = int(ys[(k+1)] - offsetY)
                cv2.line(img, (x,y), (x_succ,y_succ), color_line)
          
            for k in range(c.get_nb_landmarks()):
                x = int(xs[k] - offsetX)
                y = int(ys[k] - offsetY)
                if (k == 0):
                    img[y,x] = color_init
                elif (k == c.get_nb_landmarks()-1):
                    img[y,x] = color_end
                else:
                    img[y,x] = color_mid
                
            fname = c.get_fname_vis_ff_models(i, method)
            cv2.imwrite(fname, img) 
            
def create_all_profile_normals_images():
    '''
    Stores all the preprocessed images corresponding to all the methods used with the landmarks
    of the models (transformed to the image coordinate system) and the points along the profile
    normals marked.
    '''
    create_profile_normals_images(method='SC')
    create_profile_normals_images(method='SCD')
    create_profile_normals_images(method='EH')
    create_profile_normals_images(method='EHD') 
    
def create_profile_normals_images(color_init=np.array([0,255,255]), color_mid=np.array([255,0,255]), color_end=np.array([255,255,0]), color_line=np.array([255,0,0]), method=''):
    '''
    Stores all the preprocessed images corresponding to the given method with the landmarks
    of the models (transformed to the image coordinate system) and the points along the profile
    normals marked.
    @param color_init:  the BGR color for the first landmark 
    @param color_mid:   the BGR color for all landmarks except the first and last landmark
    @param color_end:   the BGR color for the last landmark
    @param color_line:  the BGR color for the line between two consecutive landmarks
    @param method:      the method used for preproccesing
    '''
    for i in c.get_trainingSamples_range():
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        for j in range(c.get_nb_teeth()):
            xs, ys = mu.extract_coordinates(mu.full_align_with(MS[j], XS[j,(i-1),:]))
            
            for k in range(c.get_nb_landmarks()):
                x = int(xs[k] - offsetX)
                y = int(ys[k] - offsetY)
                if (k == c.get_nb_landmarks()-1):
                    x_succ = int(xs[0] - offsetX)
                    y_succ = int(ys[0] - offsetY)
                else:
                    x_succ = int(xs[(k+1)] - offsetX)
                    y_succ = int(ys[(k+1)] - offsetY)
                cv2.line(img, (x,y), (x_succ,y_succ), color_line)
          
            for k in range(c.get_nb_landmarks()):
                x = int(xs[k] - offsetX)
                y = int(ys[k] - offsetY)
                tx, ty, nx, ny = ff.create_ricos(img, k, xs, ys)
                for n in range(-5, 5+1):
                    for t in range(-5, 5+1):
                        kx = int(x + n * nx + t * tx)
                        ky = int(y + n * ny + t * ty)
                        img[ky, kx] = np.array([0,255,0])
                
                if (k == 0):
                    img[y,x] = color_init
                elif (k == c.get_nb_landmarks()-1):
                    img[y,x] = color_end
                else:
                    img[y,x] = color_mid
                
            fname = c.get_fname_vis_ff_profile_normals(i, method)
            cv2.imwrite(fname, img) 
            
def preprocess():
    '''
    Creates XS and MS, used by the drawing functions.
        * XS contains for each tooth, for each training sample, all landmarks (in the image coordinate frame)
        * MS contains for each tooth, the tooth model (in the model coordinate frame)
    '''
    global XS, MS
    XS = l.create_full_XS()
    MS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    for j in range(c.get_nb_teeth()):
        M, Y = pa.PA(l.create_full_X(j+1))
        MS[j,:] = M
        
def create_all():
    '''
    Stores all the preprocessed images corresponding to all the methods used for all the visualizations:
        * Stores all the preprocessed images corresponding to all methods used with the landmarks
          of the training samples marked.
        * Stores all the preprocessed images corresponding to all the method used with the landmarks
          of the models (transformed to the image coordinate system) marked.
        * Stores all the preprocessed images corresponding to all the methods used with the landmarks
          of the models (transformed to the image coordinate system) and the points along the profile
          normals marked.
    '''
    create_all_landmarks_images()
    create_all_landmarks_and_models_images()
    create_all_models_images()
    create_all_profile_normals_images()

if __name__ == '__main__':
    #preprocess for visualizations
    preprocess()
    #store all visualizations
    create_all()