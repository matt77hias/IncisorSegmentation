
import numpy as np
import cv2
import configuration as c
import loader as l
import math_utils as mu
import procrustes_analysis as pa

XS = None
MS = None
offsetY = 497.0
offsetX = 1234.0

def create_all_gradient_images():
    create_gradient_images(method='SC')
    create_gradient_images(method='SCD')
    create_gradient_images(method='EH')
    create_gradient_images(method='EHD')

def create_gradient_images(method=''):
    for i in c.get_trainingSamples_range():
        fname = c.get_fname_vis_pre(i, method)
        img = cv2.imread(fname)
        temp = cv2.Scharr(img, ddepth=-1, dx=1, dy=0)
        gradient = cv2.Scharr(temp, ddepth=-1, dx=0, dy=1)
        fname = c.get_fname_vis_ff_gradients(i, method)
        cv2.imwrite(fname, gradient)
    
def create_all_landmarks_images(): 
    create_landmarks_images(method='SC')
    create_landmarks_images(method='SCD')
    create_landmarks_images(method='EH')
    create_landmarks_images(method='EHD')    
                
def create_landmarks_images(color_init=np.array([0,255,255]), color_mid=np.array([255,0,255]), color_end=np.array([255,255,0]), color_line=np.array([0,0,255]), method=''):
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
    create_landmarks_and_models_images(method='SC')
    create_landmarks_and_models_images(method='SCD')
    create_landmarks_and_models_images(method='EH')
    create_landmarks_and_models_images(method='EHD')
    
def create_landmarks_and_models_images(color_init=np.array([0,255,255]), color_mid=np.array([255,0,255]), color_end=np.array([255,255,0]), color_line=np.array([0,0,255]), color_model_line=np.array([255,0,0]), method=''):
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
    create_models_images(method='SC')
    create_models_images(method='SCD')
    create_models_images(method='EH')
    create_models_images(method='EHD') 
    
def create_models_images(color_init=np.array([0,255,255]), color_mid=np.array([255,0,255]), color_end=np.array([255,255,0]), color_line=np.array([255,0,0]), method=''):
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
            
def preprocess():
    global XS, MS
    XS = l.create_full_XS()
    MS = np.zeros((c.get_nb_teeth(), c.get_nb_dim()))
    for j in range(c.get_nb_teeth()):
        M, Y = pa.PA(l.create_full_X(j+1))
        MS[j,:] = M

if __name__ == '__main__':
    preprocess()
    create_all_landmarks_and_models_images()